import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define lootbox contents and probabilities
lootbox_items = {
    "Common": 0.60,
    "Rare": 0.30,
    "Epic": 0.09,
    "Legendary": 0.01
}

def open_lootbox(pity_counter):
    rand = np.random.random()
    cumulative_prob = 0
    for rarity, prob in lootbox_items.items():
        cumulative_prob += prob
        if rand <= cumulative_prob or (rarity == "Legendary" and pity_counter >= 49):
            return rarity
    return "Common"  # Fallback, should never reach here

def simulate_lootboxes(num_boxes):
    results = {"Common": 0, "Rare": 0, "Epic": 0, "Legendary": 0}
    pity_counter = 0
    for _ in range(num_boxes):
        item = open_lootbox(pity_counter)
        results[item] += 1
        if item == "Legendary":
            pity_counter = 0
        else:
            pity_counter += 1
    return results

def show_lootbox_simulation():
    st.title("Lootbox Simulation")
    st.write("This simulation demonstrates the odds of obtaining items from lootboxes.")

    # User interface
    st.subheader("Lootbox Simulation")
    num_boxes = st.slider("Number of lootboxes to open", 1, 1000, 100)

    if st.button("Run Simulation"):
        results = simulate_lootboxes(num_boxes)
        
        # Display results
        st.write("Simulation Results:")
        for rarity, count in results.items():
            st.write(f"{rarity}: {count} ({count/num_boxes*100:.2f}%)")
        
        # Create a pie chart
        fig, ax = plt.subplots()
        ax.pie(results.values(), labels=results.keys(), autopct='%1.1f%%')
        ax.set_title("Lootbox Rewards Distribution")
        st.pyplot(fig)

        # Create a bar chart for comparison with expected probabilities
        expected_results = {rarity: prob * num_boxes for rarity, prob in lootbox_items.items()}
        comparison_df = pd.DataFrame({
            'Rarity': results.keys(),
            'Simulated': results.values(),
            'Expected': expected_results.values()
        })
        
        fig, ax = plt.subplots()
        comparison_df.plot(x='Rarity', y=['Simulated', 'Expected'], kind='bar', ax=ax)
        ax.set_title("Simulated vs Expected Results")
        ax.set_ylabel("Number of items")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Display theoretical probabilities
    st.subheader("Theoretical Probabilities")
    for rarity, prob in lootbox_items.items():
        st.write(f"{rarity}: {prob*100:.2f}%")

    st.write("Note: There's a guaranteed Legendary item every 50 openings if one hasn't been obtained.")

    # Expected value calculation
    st.subheader("Expected Value Calculator")
    st.write("Assign monetary values to each rarity to calculate the expected value of a lootbox.")
    
    rarity_values = {}
    for rarity in lootbox_items.keys():
        rarity_values[rarity] = st.number_input(f"Value of {rarity} item", value=0.0, step=0.1)
    
    expected_value = sum(prob * rarity_values[rarity] for rarity, prob in lootbox_items.items())
    st.write(f"Expected value of one lootbox: ${expected_value:.2f}")

    lootbox_cost = st.number_input("Cost of one lootbox", value=1.0, step=0.1)
    roi = (expected_value - lootbox_cost) / lootbox_cost * 100
    st.write(f"Return on Investment: {roi:.2f}%")

    if roi > 0:
        st.write("The expected value is higher than the cost. This lootbox system is favorable to the player in the long run.")
    else:
        st.write("The expected value is lower than the cost. This lootbox system is favorable to the house in the long run.")