import * as tf from '@tensorflow/tfjs';
import config from './config';

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

/**
 * a generator that generates the next character each time you call it,
 * and never has a done state. So if you wanted that, sucks to be you
 * (don't kill me plase)
 *
 * @param {tf.LayersModel} model the model to actually generate the
 *  characters with
 * @param {string} startString the string to start the model with
 */
function* generateText(model, startString) {
  let inputEval = startString.split('').map((c) => char2idx[c]);
  inputEval = tf.expandDims(inputEval, 0);

  model.resetStates();
  while (true) {
    let predictions = model.predict(inputEval);
    predictions = tf.squeeze(predictions, 0);

    const predId = sample(tf.squeeze(predictions), config.temperature);

    inputEval = tf.expandDims([predId], 0);

    yield idx2char[predId];
  }
}

const { ids, messages } = config;

const pre = document.querySelector(`#${ids.outputPre}`);
const seed = document.querySelector(`#${ids.seedInput}`);
const generate = document.querySelector(`#${ids.generateButton}`);
generate.value = messages.start;

// true if text is currently generating, false otherwise
let interval = -1;
generate.addEventListener('click', async () => {
  if (interval === -1) {
    interval = -2;
    const model = await tf.loadLayersModel('http://localhost:9000/model/model.json');
    const startString = seed.value || ' ';
    let generated = startString;
    // just ignore the fact that I moved it lowercase because otherwise it dies
    const generator = generateText(model, startString.toLowerCase());
    interval = setInterval(() => {
      generated += generator.next().value;
      pre.textContent = generated;
    }, 0);
    generate.value = messages.stop;
  } else if (interval !== -2) {
    clearInterval(interval);
    interval = -1;
    generate.value = messages.start;
  }
});
