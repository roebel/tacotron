#! /usr/bin/env python
import argparse
from datetime import datetime
import math
import os
import sys
import subprocess
import time
import tensorflow as tf
import traceback
import manage_gpus as gpl

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
log = infolog.log

def get_gpu_lock(gpu_device_id, soft=False):
  gpu_id_locked=gpl.obtain_lock_id(id=gpu_device_id, hard=not soft)
  if gpu_id_locked < 0:
    #  lock removal has time delay of 2 so be sure to have the lock of the last run removed we wait
    # for 3 s here
    time.sleep(3)
    gpu_id_locked=gpl.obtain_lock_id(id=gpu_device_id, hard=not soft)
    if gpu_id_locked < 0:
      if gpu_device_id < 0:
        raise RuntimeError("No GPUs available for locking")
      else:
        raise RuntimeError("cannot obtain the selected GPU {0}".format(str(gpu_device_id)))
      
    return gpu_id_locked

def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  # Set up DataFeeder:
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = create_model(args.model, hparams)
    model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  # Train!
  with tf.Session() as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      if args.restore_step:
        # Restore from a checkpoint if the user requested it.
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
      else:
        log('Starting new training run at commit: %s' % commit, slack=True)

      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()
        step, loss, opt = sess.run([global_step, model.loss, model.optimize])
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
          step, time_window.average, loss, loss_window.average)
        log(message, slack=(step % args.checkpoint_interval == 0))

        if loss > 100 or math.isnan(loss):
          log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
          raise Exception('Loss Exploded')

        if step % args.summary_interval == 0:
          log('Writing summary at step: %d' % step)
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          log('Saving audio and alignment...')
          input_seq, spectrogram, alignment = sess.run([
            model.inputs[0], model.linear_outputs[0], model.alignments[0]])
          waveform = audio.inv_spectrogram(spectrogram.T)
          audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
          plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
            info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
          log('Input: %s' % sequence_to_text(input_seq))

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.dirname(os.path.abspath(__file__)),
                      help="tacotron install-dir (Def: %(default)s)")
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  device_arg = parser.add_mutually_exclusive_group()
  device_arg.add_argument("--cpu", action="store_true",
                          help='use cpu for calculations, this is the default on sytems without available gpu card (Def: %(default)s)')
  device_arg.add_argument('-d',"--gpu_device", default=None, nargs="?", const=-1, type=int,
                          help='use gpu device, use without argument for arbitrary gpu, this is the default for systems with gpu (Def: %(default)s)')

  parser.add_argument("--soft_lock", action="store_true",
                      help='only request a soft lock on the GPU (Def: %(default)s)')

  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)

  comp_device = None
  gpu_device_id = None
  if not args.cpu :
    # gpu_ids will be None on systems without gpu nvidia card
    gpu_ids=gpl.board_ids()
    if gpu_ids is not None:
      if args.gpu_device is None or args.gpu_device == -1:
        gpu_device_id = -1
      elif args.gpu_device in gpu_ids:
        gpu_device_id = args.gpu_device
      else:
        raise RuntimeError("train_onsets::error:: selected gpu device if {} is not free, select an id from {}".format(args.gpu_device, gpu_ids) )
    elif args.gpu_device is not None:
      raise RuntimeError("train_onsets::error:: no gpu devices available on thsi system, you cannot select a gpu")

    print("gpu_device_id", gpu_device_id)  
    try:
      # now we lock a GPU because we will need one
      if gpu_device_id is not None:
        gpu_id_locked = get_gpu_lock(gpu_device_id = gpu_device_id, soft=args.soft_lock)
        # obtainlock positions CUDA_VISIBLE_DEVICES such that only the selected GPU is visibale,
        # therefore we need now select /GPU:0
        comp_device = "/GPU:0"
      else:
        gpu_id_locked=-1
        comp_device="/cpu:0"
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

      run_name = args.name or args.model
      log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
      os.makedirs(log_dir, exist_ok=True)
      infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
      hparams.parse(args.hparams)
      train(log_dir, args)
    except Exception as ex:
      import traceback
      tb=traceback.format_exc()
      print("{0} received exception::".format(sys.argv[0]), str(ex), tb, file=sys.stderr)

    finally:
      # terminate input pipeline
      if ("GPU" in comp_device) and (gpu_id_locked >= 0):
        gpl.free_lock(gpu_id_locked)
                             
if __name__ == '__main__':
  main()
