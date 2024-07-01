from transformers import pipeline

# Utiliser GPT-3 pour la génération de texte
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# def generate_response(prompt):
    # response = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
    # return response[0]['generated_text']

# Utiliser un modèle pré-entraîné léger
generator = pipeline('text-generation', model='distilgpt2')

def generate_response(prompt):
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    print("Bienvenue! Tapez 'quitter' pour terminer la conversation.")
    
    while True:
        user_input = input("Vous: ")
        
        if "quitter" in user_input.lower():
            print("Au revoir!")
            break
        
        response = generate_response(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
