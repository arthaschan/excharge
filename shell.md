
tmux list-sessions
tmux switch -t 0

conda activate hermes
hermes chat --model claude-opus-4-8

hermes chat 
hermes chat --model Qwen/Qwen3-32B --provider qwen-local
vllm_env
start_vllm
vllm serve /home/student/models/Qwen3-32B \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
  
stop_vllm
hermes chat --model "/home/student/models/Qwen3-32B" --provider qwen-local

📁 Your files:

   Config:    /home/student/.hermes/config.yaml
   API Keys:  /home/student/.hermes/.env
   Data:      /home/student/.hermes/cron/, sessions/, logs/
   Code:      /home/student/.hermes/hermes-agent

─────────────────────────────────────────────────────────

🚀 Commands:

   hermes              Start chatting
   hermes setup        Configure API keys & settings
   hermes config       View/edit configuration
   hermes config edit  Open config in editor
   hermes gateway install Install gateway service (messaging + cron)
   hermes update       Update to latest version

─────────────────────────────────────────────────────────

⚡ Reload your shell to use 'hermes' command:

   source ~/.bashrc