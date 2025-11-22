const { spawn } = require('child_process');
const path = require('path');
const { REPO_ROOT } = require('../request');

function createLocalProcessBackend(config = {}) {
  const pythonBin = config.pythonBin || process.env.PYTHON_BIN || 'python3';
  const trainScript =
    config.trainScript ||
    (process.env.TRAIN_SCRIPT_PATH
      ? path.resolve(process.env.TRAIN_SCRIPT_PATH)
      : path.join(REPO_ROOT, 'training', 'common', 'train.py'));

  return {
    name: 'local',
    submit: ({ cliArgs, hfToken, env = {}, helpers }) => {
      const command = [pythonBin, trainScript, ...cliArgs];
      const childEnv = { ...process.env, ...env };
      if (hfToken) {
        childEnv.HF_TOKEN = hfToken;
      }

      helpers.setCommand(command.join(' '));
      helpers.setStatus('running');

      const child = spawn(command[0], command.slice(1), {
        cwd: REPO_ROOT,
        env: childEnv,
      });

      child.stdout.on('data', (data) => helpers.appendLog('stdout', data));
      child.stderr.on('data', (data) => helpers.appendLog('stderr', data));

      child.on('error', (err) => {
        helpers.appendLog('stderr', err.message);
        helpers.setError(err.message);
        helpers.finish('failed');
      });

      child.on('close', (code) => {
        helpers.setExitCode(code);
        if (code === 0) {
          helpers.finish('succeeded');
        } else {
          helpers.setError(
            helpers.getError() ||
              `Training process exited with code ${code}`
          );
          helpers.finish('failed');
        }
      });
    },
  };
}

module.exports = { createLocalProcessBackend };
