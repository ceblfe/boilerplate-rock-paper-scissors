
import random
import numpy as np
import tensorflow as tf

counter = {'R': 'P', 'P': 'S', 'S': 'R'}

def move_to_onehot(move):
    if move == 'R':
        return [1, 0, 0]
    elif move == 'P':
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def get_state(opponent_history, my_history, length=5):
    # Pad with initial moves if history is shorter than length
    opp_padded = ['R'] * (length - len(opponent_history)) + opponent_history[-length:] if len(opponent_history) < length else opponent_history[-length:]
    my_padded = ['R'] * (length - len(my_history)) + my_history[-length:] if len(my_history) < length else my_history[-length:]
    state = []
    for m in opp_padded:
        state += move_to_onehot(m)
    for m in my_padded:
        state += move_to_onehot(m)
    return np.array(state).reshape(1, -1)  # Shape (1, 30)

def create_model():
    inputs = tf.keras.Input(shape=(30,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

training_buffer = []

def simulate_quincy(length):
    counter_local = [0]
    choices = ["R", "R", "P", "P", "S"]
    sim_opp = []
    for _ in range(length):
        counter_local[0] += 1
        sim_move = choices[counter_local[0] % 5]
        sim_opp.append(sim_move)
    return sim_opp

def detect_quincy(opponent_history):
    if len(opponent_history) < 10:
        return False
    sim = simulate_quincy(len(opponent_history))
    return sim == opponent_history

def quincy_strategy(opponent_history):
    counter_local = [len(opponent_history)]
    choices = ["R", "R", "P", "P", "S"]
    next_idx = (counter_local[0]) % 5
    predicted = choices[next_idx]
    return counter[predicted]

def simulate_kris(my_history):
    sim_opp = []
    for i in range(len(my_history)):
        prev = "R" if i == 0 else my_history[i - 1]
        sim_move = counter[prev]
        sim_opp.append(sim_move)
    return sim_opp

def detect_kris(opponent_history, my_history):
    if len(opponent_history) < 5 or len(my_history) < 5:
        return False
    sim = simulate_kris(my_history)
    return sim == opponent_history

def kris_strategy(my_history):
    if not my_history:
        return random.choice(['R', 'P', 'S'])
    my_last = my_history[-1]
    kris_next = counter[my_last]
    return counter[kris_next]

def simulate_mrugesh(my_history):
    sim_opp = []
    sim_opp_history = []  # Mrugesh's opponent_history = player's moves
    for i in range(len(my_history)):
        prev = my_history[i] if i > 0 else ''
        sim_opp_history.append(prev)
        last_ten = sim_opp_history[-10:]
        most_frequent = max(set(last_ten), key=last_ten.count) if last_ten else "S"
        sim_move = counter.get(most_frequent, 'R')
        sim_opp.append(sim_move)
    return sim_opp

def detect_mrugesh(opponent_history, my_history):
    if len(opponent_history) < 5 or len(my_history) < 5:
        return False
    sim = simulate_mrugesh(my_history)
    return sim == opponent_history

def mrugesh_strategy(my_history):
    last_ten = my_history[-10:]
    most_frequent = max(set(last_ten), key=last_ten.count) if last_ten else "S"
    mrugesh_next = counter.get(most_frequent, 'R')
    return counter[mrugesh_next]

def simulate_abbey(my_history):
    sim_opp = []
    play_order = [{"RR": 0, "RP": 0, "RS": 0, "PR": 0, "PP": 0, "PS": 0, "SR": 0, "SP": 0, "SS": 0}]
    sim_opp_history = []  # Abbey's opponent_history = player's moves
    for i in range(len(my_history)):
        prev = '' if i == 0 else my_history[i - 1]
        if not prev:
            prev = 'R'
        sim_opp_history.append(prev)
        last_two = "".join(sim_opp_history[-2:])
        if len(last_two) == 2:
            play_order[0][last_two] += 1
        potential_plays = [
            prev + "R",
            prev + "P",
            prev + "S",
        ]
        sub_order = {
            k: play_order[0][k]
            for k in potential_plays if k in play_order[0]
        }
        if sub_order:
            prediction = max(sub_order, key=sub_order.get)[-1]
        else:
            prediction = 'R'
        ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
        sim_move = ideal_response[prediction]
        sim_opp.append(sim_move)
    return sim_opp

def detect_abbey(opponent_history, my_history):
    if len(opponent_history) < 10 or len(my_history) < 10:
        return False
    sim = simulate_abbey(my_history)
    return sim == opponent_history

def abbey_strategy(my_history):
    if len(my_history) < 2:
        return random.choice(['R', 'P', 'S'])
    play_order = {"RR": 0, "RP": 0, "RS": 0, "PR": 0, "PP": 0, "PS": 0, "SR": 0, "SP": 0, "SS": 0}
    for i in range(1, len(my_history)):
        last_two = ''.join(my_history[i-1:i+1])
        if len(last_two) == 2 and last_two in play_order:
            play_order[last_two] += 1
    last = my_history[-1]
    potential = [last + "R", last + "P", last + "S"]
    sub_order = {k: play_order.get(k, 0) for k in potential}
    prediction = max(sub_order, key=sub_order.get)[-1] if any(sub_order.values()) else 'R'
    abbey_next = counter[prediction]
    return counter[abbey_next]

def player(prev_play, opponent_history=[], my_history=[]):
    global model, training_buffer
    if not prev_play:
        opponent_history.clear()
        my_history.clear()
        training_buffer.clear()
        model = create_model()
        simple_pretrain(50)  # Quick pre-training per mactch
    if prev_play != "":
        opponent_history.append(prev_play)
    
    # Train with the previous observation
    if len(opponent_history) > 1 and len(my_history) > 0:
        previous_opp_history = opponent_history[:-1]
        previous_my_history = my_history[:]
        state = get_state(previous_opp_history, previous_my_history)
        label = move_to_onehot(prev_play)
        training_buffer.append((state, label))
        if len(training_buffer) >= 10:  # Small buffer for more frequent learning
            X = np.vstack([s for s, l in training_buffer])
            y = np.array([l for s, l in training_buffer])
            model.fit(X, y, epochs=1, verbose=0, batch_size=len(training_buffer))
            training_buffer = []
    
    # Detects the bot and applies specific strategy
    if detect_quincy(opponent_history):
        guess = quincy_strategy(opponent_history)
    elif detect_kris(opponent_history, my_history):
        guess = kris_strategy(my_history)
    elif detect_mrugesh(opponent_history, my_history):
        guess = mrugesh_strategy(my_history)
    elif detect_abbey(opponent_history, my_history):
        guess = abbey_strategy(my_history)
    else:
        if not opponent_history:
            guess = random.choice(['R', 'P', 'S'])
        else:
            state = get_state(opponent_history, my_history)
            probs = model.predict(state, verbose=0)[0]
            predicted_idx = np.argmax(probs)
            moves = ['R', 'P', 'S']
            predicted_opp = moves[predicted_idx]
            guess = counter[predicted_opp]
    
    # Save the movement in my_history
    my_history.append(guess)
    return guess

# Simpe pre-training with random data to give the model a starting point
def simple_pretrain(num_samples=50):
    global training_buffer, model
    training_buffer.clear()
    for _ in range(num_samples):
        # Generates random sequences of opponent and player for pre-training
        random_opp_history = [random.choice(['R', 'P', 'S']) for _ in range(6)]  # 5 prev + 1 next
        random_my_history = [random.choice(['R', 'P', 'S']) for _ in range(5)]
        state = get_state(random_opp_history[:-1], random_my_history)
        label = move_to_onehot(random_opp_history[-1])
        training_buffer.append((state, label))
    if training_buffer:
        X = np.vstack([s for s, l in training_buffer])
        y = np.array([l for s, l in training_buffer])
        model.fit(X, y, epochs=2, verbose=0)
        training_buffer.clear()

# Performs simple pre-training when loading the module
simple_pretrain(50)