import subprocess

class TinyLlamaHandler:
    def __init__(self):
        self.process = subprocess.Popen(
            ["ollama", "run", "tinyllama"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def send_query(self, query, timeout=10):
        try:
            # Send the query
            self.process.stdin.write(query + "\n")
            self.process.stdin.flush()

            # Read response with timeout
            response_lines = []
            while True:
                line = self.process.stdout.readline().strip()
                if line == ">>>":  # End of response
                    break
                response_lines.append(line)

            return "\n".join(response_lines).strip()
        except subprocess.TimeoutExpired:
            print("TinyLlama response timeout.")
            return "I'm unable to process that right now."
        except Exception as e:
            print(f"Error communicating with TinyLlama: {e}")
            return "I'm unable to process that right now."

    def close(self):
        if self.process:
            self.process.stdin.write("/bye\n")
            self.process.stdin.flush()
            self.process.terminate()
