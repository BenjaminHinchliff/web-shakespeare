import * as tf from '@tensorflow/tfjs';

const vocabString = '\n !$&\',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
const vocab = [...new Set(vocabString.split(''))].sort();
const char2idx = vocab.reduce((p, c, i) => ({ ...p, [c]: i }), {});
const idx2char = vocab;

/**
 * Draw a sample based on probabilities. Why doesn't tf.random.categoriacl exist?
 *
 * @param {tf.Tensor} probs
 * @param {tf.Tensor} temperature
 * @returns {number}
 */
function sample(probs, temperature) {
  return tf.tidy(() => {
    const logits = tf.div(probs, Math.max(temperature, 1e-6));
    const isNormalized = false;
    return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
  });
}

const TEMPERATURE = 1.0;
/**
 * a generator that generates the next character each time you call it,
 * and never has a done state. So if you wanted that, sucks to be you
 * (don't kill me plase)
 *
 * @param {tf.LayersModel} model the model to actually genearte the characters with
 * @param {string} startString the string to start the model with
 */
function* generateText(model, startString) {
  let inputEval = startString.split('').map((c) => char2idx[c]);
  inputEval = tf.expandDims(inputEval, 0);

  model.resetStates();
  while (true) {
    let predictions = model.predict(inputEval);
    predictions = tf.squeeze(predictions, 0);

    const predId = sample(tf.squeeze(predictions), TEMPERATURE);

    inputEval = tf.expandDims([predId], 0);

    yield idx2char[predId];
  }
}

const pre = document.createElement('pre');
document.body.appendChild(pre);

const NUM_CHARS_GENERATE = 1000;
tf.loadLayersModel('http://localhost:9000/model/model.json').then((model) => {
  const startString = idx2char[Math.floor(Math.random() * idx2char.length)].toLowerCase();
  let generated = startString;
  const generator = generateText(model, startString);
  const interval = setInterval(() => {
    if (generated.length < NUM_CHARS_GENERATE || generated.slice(-1) !== '\n') {
      generated += generator.next().value;
      pre.textContent = generated;
    } else {
      clearInterval(interval);
    }
  },
  0);
});
