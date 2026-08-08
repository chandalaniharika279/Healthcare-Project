def optimize_symptoms(conversation_text, dataset_df):
    """
    Extract clinically meaningful symptoms from conversation
    and align them with dataset-supported symptoms.
    """
    text = conversation_text.lower()

    dataset_symptoms = (
        dataset_df["symptom_text"]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    matched_symptoms = []

    for s in dataset_symptoms:
        if all(word in text for word in s.split()):
            matched_symptoms.append(s)

    # remove duplicates and sort by length (more specific first)
    matched_symptoms = sorted(set(matched_symptoms), key=len, reverse=True)

    # split into primary / secondary
    primary = matched_symptoms[:3]
    secondary = matched_symptoms[3:6]

    return primary, secondary
