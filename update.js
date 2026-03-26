module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt"
      }
    }
  ]
}
