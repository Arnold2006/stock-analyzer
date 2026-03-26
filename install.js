module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "python -m venv venv"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "venv/bin/pip install --upgrade pip"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "venv/bin/pip install -r requirements.txt"
      }
    }
  ]
}
