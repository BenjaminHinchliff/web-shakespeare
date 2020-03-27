const config = {
  ids: {
    generateButton: 'generate',
    seedInput: 'seed',
    outputPre: 'output',
  },
  messages: {
    stop: 'stop!',
    start: 'generate!',
  },
  locations: {
    model: 'model/model.json',
  },
  temperature: 0.9, // temprature of the RNN (changes craziness - higher is more)
};

export default config;
