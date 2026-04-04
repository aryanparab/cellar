import os
import json
from agent.data import load_dataset, query_wines
from agent.agent import ask_agent, ask_barkeep, sanitize_for_json

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # 1. Initialize Data
    print("🍷 Loading Wine Cellar...")
    load_dataset() # Loads from Google Sheets by default
    
    history = []
    selected_wine = None

    while True:
        print("\n" + "="*50)
        if selected_wine:
            print(f"📍 MODE: BARKEEP | FOCUS: {selected_wine.get('name')}")
        else:
            print("📍 MODE: GLOBAL SOMMELIER (Ask about any wine, region, or price)")
        print("="*50)
        
        user_input = input("\nYou (type 'exit' to quit, 'clear' to reset): ").strip()

        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.lower() == 'clear':
            history = []
            selected_wine = None
            clear_screen()
            continue
        if not user_input:
            continue

        # --- EXECUTION LOGIC ---
        
        if selected_wine:
            # BARKEEP MODE: Specific wine context + History
            response = ask_barkeep(user_input, selected_wine, history)
            print(f"\nBarkeep: {response}")
        else:
            # GLOBAL MODE: Tool-based search
            response = ask_agent(user_input)
            print(f"\nSommelier: {response}")

        # Update History
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # --- SPECIAL CLI COMMAND: SELECT A WINE ---
        # In a real UI, you click a wine. Here, we'll let you "pick" one 
        # if the agent just recommended some.
        if "pick" in user_input.lower() and not selected_wine:
            print("\nSearching for wine to focus on...")
            search = user_input.lower().replace("pick", "").strip()
            results = query_wines(name_contains=search, limit=1)
            if results:
                selected_wine = sanitize_for_json(results[0])
                print(f"✅ Now focusing on: {selected_wine['name']}")
            else:
                print("❌ Could not find that wine in the list.")

if __name__ == "__main__":
    main()