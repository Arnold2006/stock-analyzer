module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "venv/bin/python app.py",
        on: [{ event: "/.*/", done: true }]
      }
    }
  ]
}
