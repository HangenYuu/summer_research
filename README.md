> Models to try: https://www.one-tab.com/page/Aof0mJwYQ-yCVMKN9x56gQ
# Preliminary model selection
## Run 1 - focus position is ~3250
- GPT-4.1: The best so far. I found the correct position after 6 iterations.
- Qwen2.5VL-32B: It was getting near the answer, but it took too many iterations, and the context window maxxed out. It stopped at ~2985 (iteration 14).
- Llama 4 Maverick: It did not work. It got lost around position 750. To be fair, Groq free API is not the best.
## Run 2 - focus position is ~3250
- Gemma3-12B-Q4: It got off in a good start comparable to GPT-4.1. But it got lost around 3500, repeatedly overshot and undershot. It also could not distinguish between "over" and "under".
- Gemma3-27B-Q4: I hoped that it would be better. Performance was similar to Gemma3-12B. It stopped early at around 2260.
- SmolVLM2-2.2B-16F: It output gibberish to me.
- MiniCPM4-8B-Q8: It also output gibberish to me.
- MedGemma-4B-Q8: It output errors to me. There is not even chat history to save.
- Gemini 2.5 Flash Preview 20/05: It stoppped at 3010. Potential model 🙂.

**Winner**: GPT-4.1 (no surprise)

**Note**: Ollama was run on a 12GB Nvidia GeForce RTX 4070 that came to NTU some months ago. I tried to use the quantized version that fits as much possible in GPU.
