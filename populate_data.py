def populate_initial_data():
    sentences = [
        "A man at the grocery store wearing black pants and a red shirt.",
        "A girl on her bike in the park with a white dog and a child.",
        "An elderly woman with a cane standing near the bus stop.",
        "A teenager skateboarding down the street with headphones on.",
        "A child flying a kite in the open field, wearing a yellow cap.",
        "A couple jogging together along the river in matching outfits.",
        "A businessman in a suit talking on his phone, looking hurried.",
        "A musician playing the guitar on the sidewalk for tips.",
        "A group of kids playing soccer in the park.",
        "A woman pushing a stroller while talking on her phone.",
        "A man reading a newspaper on a bench with a coffee by his side.",
        "A woman selling flowers at a small roadside stall.",
        "A police officer directing traffic at a busy intersection.",
        "A food vendor making hot dogs at his cart.",
        "A cyclist riding against traffic, wearing a bright orange vest.",
        "A painter on a ladder, working on a mural on the side of a building.",
        "A mail carrier delivering packages, pushing a cart.",
        "A student with a heavy backpack waiting for the school bus.",
        "A dog walker managing several dogs of different sizes.",
        "A person handing out flyers for a local event."
    ]
    for i, sentence in enumerate(sentences):
        unique_id = str(i + 1)
        add_to_faiss(unique_id, sentence)
       