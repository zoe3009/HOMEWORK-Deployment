
const REG_ONNX_PATH = "./zia_regression_winewhite.onnx";
const CLS_ONNX_PATH = "./zia_classification_iris.onnx";


const N_FEATURES_REG = 11;  // Wine White features
const N_FEATURES_CLS = 4;   // Iris features
const N_CLASSES_CLS  = 3;   // setosa, versicolor, virginica

const IRIS_CLASS_NAMES = ["setosa", "versicolor", "virginica"];


const MEANS_REG = [6.865045941807049, 0.2793376722817761, 0.33273098519652555, 6.450701888718721, 0.04573404798366544, 35.09456355283308, 138.0011485451761, 0.9940706457376164, 3.189293006636035, 0.4897805002552349, 10.508840394759261];

const SCALES_REG = [0.8443753700979084, 0.10159292331295482, 0.1197423412775926, 5.138654709456886, 0.021794640388589484, 16.674829911539792, 42.06229805927688, 0.0030213035674401863, 0.15016406004141622, 0.11357548286882742, 1.2277299542941176];


const MEANS_CLS = [5.841666666666668, 3.0483333333333342, 3.769999999999999, 1.2049999999999987];

const SCALES_CLS = [0.837415003978845, 0.44665111913239636, 1.7611359970201048, 0.7594789880788891];

function parseInputText(inputId) {
  const raw = document.getElementById(inputId).value.trim();
  if (!raw) {
    throw new Error("Input is empty.");
  }
  const parts = raw.split(",").map(v => v.trim()).filter(v => v.length > 0);
  const nums = parts.map(v => Number(v));
  if (nums.some(v => Number.isNaN(v))) {
    throw new Error("All inputs must be valid numbers.");
  }
  return nums;
}

function standardizeFeatures(rawFeatures, means, scales) {
  if (rawFeatures.length !== means.length) {
    throw new Error(`Expected ${means.length} features but got ${rawFeatures.length}.`);
  }
  const scaled = [];
  for (let i = 0; i < rawFeatures.length; i++) {
    const v = (rawFeatures[i] - means[i]) / (scales[i] || 1e-8);
    scaled.push(v);
  }
  return scaled;
}

async function createSession(path) {
  return await ort.InferenceSession.create(path);
}

// --------- REGRESSION: Wine White ---------
let regSession = null;

async function runZiaRegression() {
  try {
    if (!regSession) {
      regSession = await createSession(REG_ONNX_PATH);
    }

    // 1) Read and parse 11 raw features
    const rawFeatures = parseInputText("reg_inputs");
    if (rawFeatures.length !== N_FEATURES_REG) {
      throw new Error(`Regression expects ${N_FEATURES_REG} features.`);
    }

    // 2) Standardize using MEANS_REG and SCALES_REG
    const scaled = standardizeFeatures(rawFeatures, MEANS_REG, SCALES_REG);

    // 3) Create tensor [1, 11]
    const data = Float32Array.from(scaled);
    const tensor = new ort.Tensor("float32", data, [1, N_FEATURES_REG]);

    // 4) Run ONNX
    const feeds = { input: tensor };
    const results = await regSession.run(feeds);

    // 5) Extract prediction
    const pred = results.output.data[0];

    document.getElementById("reg_output").textContent = pred.toFixed(3);
  } catch (err) {
    console.error(err);
    document.getElementById("reg_output").textContent = "Error: " + err.message;
  }
}

// --------- CLASSIFICATION: Iris ---------
let clsSession = null;

async function runZiaClassification() {
  try {
    if (!clsSession) {
      clsSession = await createSession(CLS_ONNX_PATH);
    }

    // 1) Read and parse 4 raw features
    const rawFeatures = parseInputText("cls_inputs");
    if (rawFeatures.length !== N_FEATURES_CLS) {
      throw new Error(`Classification expects ${N_FEATURES_CLS} features.`);
    }

    // 2) Standardize using MEANS_CLS and SCALES_CLS
    const scaled = standardizeFeatures(rawFeatures, MEANS_CLS, SCALES_CLS);

    // 3) Create tensor [1, 4]
    const data = Float32Array.from(scaled);
    const tensor = new ort.Tensor("float32", data, [1, N_FEATURES_CLS]);

    // 4) Run ONNX
    const feeds = { input: tensor };
    const results = await clsSession.run(feeds);

    // 5) Output is logits [1, 3]
    const logits = results.output.data;  // Float32Array length 3

    // Softmax
    const exps = Array.from(logits).map(x => Math.exp(x));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / (sumExp || 1e-8));

    // Argmax
    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    const className = IRIS_CLASS_NAMES[maxIdx] || `Class ${maxIdx}`;
    const probStr = probs[maxIdx].toFixed(3);

    document.getElementById("cls_output").innerHTML =
      `${className} (class ${maxIdx}, prob = ${probStr})`;
  } catch (err) {
    console.error(err);
    document.getElementById("cls_output").textContent = "Error: " + err.message;
  }
}
