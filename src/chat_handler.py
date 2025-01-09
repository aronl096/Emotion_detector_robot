import ollama

class TinyLlamaHandler:
    def __init__(self, model="tinyllama:latest"):
        """Initialize the TinyLlama handler with the specified model."""
        print("Initializing TinyLlama...")
        self.model = model
        print(f"Using model: {self.model}")
        print("TinyLlama initialized and ready.")

    def send_query(self, query):
        try:
            # Use the ollama library to generate a response
            print(f"Sending query: {query}")
            response = ollama.generate(model=self.model, prompt=query)

            if not response or not response.get("response"):
                print("No response received from TinyLlama.")
                return "I'm unable to process that right now."

            final_response = response["response"].strip()
            print(f"Final response: {final_response}")

            # Ensure the response is concise
            # concise_response = final_response.split(".")[0] + "." if "." in final_response else final_response[:400]
            # return concise_response
            return final_response

        except Exception as e:
            print(f"Error communicating with TinyLlama: {e}")
            return "I'm unable to process that right now."

    def close(self):
        """Perform cleanup if necessary."""
        print("TinyLlama session closed.")
