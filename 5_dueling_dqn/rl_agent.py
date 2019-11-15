
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Subtract, Add
from tensorflow.keras.optimizers import Adam
import keras.backend as K


# Deep Q Network off-policy
class DQNAgent:
    def __init__(self,
                 action_size,
                 state_size,
                 learning_rate=0.001,
                 discount_rate=0.95,
                 epsilon=1.0,
                 epsilon_decay=0.99,
                 epsilon_min=0.01,
                 memory_size=2000,
                 batch_size=32,
    ):
        self.action_size = action_size
        self.state_size = state_size
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size

        # total learn step
        self.learn_step_counter = 0
        # build 2 network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        inputs = Input(shape=(self.state_size, ))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        value = Dense(1, activation='relu')(x)
        a = Dense(self.action_size, activation='relu')(x)
        meam = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, meam])
        q = Add()([value, advantage])
        model = Model(inputs=inputs, outputs=q)
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def update_target_model(self):
        # copy weights from model to target model
        self.target_model.set_weights(self.model.get_weights())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def learn(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                # target[0][action] = reward + self.gamma * np.amax(t)
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
