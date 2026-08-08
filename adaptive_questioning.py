MAX_QUESTIONS = 8

def next_question(matched_df, asked):
    if len(asked) >= MAX_QUESTIONS:
        return None

    symptoms = (
        matched_df["symptom_text"]
        .astype(str)
        .str.lower()
        .str.split()
        .explode()
        .unique()
    )

    for s in symptoms:
        if s not in asked:
            return f"Do you have {s}?"
    return None
