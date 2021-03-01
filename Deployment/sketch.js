let w, t, video;
let imgArr = []
let task;

let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "U";
let result;

function preload(){
  w = loadImage("img/w.jpeg");
  t = loadImage("img/t.png");
  u = loadImage("img/u.jpg");
  task = t;
  result = t;
  imgArr.push(w);
  imgArr.push(t);
  imgArr.push(u);
}

function setup() {
  createCanvas(800, 800);
  video = createCapture(VIDEO);
  video.hide();
  setInterval(function () {
  task = imgArr[Math.floor(Math.random() * imgArr.length)];
  }, 2000);
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  const modelInfo = {
    model: 'model.json',
    metadata: 'model_meta.json',
    weights: 'model.weights.bin',
  };
  brain.load(modelInfo, brainLoaded);
}

function brainLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) { 
  
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label.toUpperCase();
  }
  //console.log(results[0].confidence); 
  classifyPose();
}

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}

function modelLoaded() {
  console.log('poseNet ready');
}


function draw() { 
  background(220); 
  image(video, 0, 0, 800, 400) 
  textSize(32); 
  text('Make this Character:', 30, 440); 
  text('Result of the Model:', 430, 440); 
  image(task, 0, 450, 400, 400); 
  if (poseLabel == 'U') { 
    result = u; 
  } 
  else if(poseLabel == 'T') { 
    result = t; 
  } 
  else { 
    result = w; 
  } 
  console.log(result); 
  console.log(poseLabel); 
  image(result, 400, 450, 400, 400); 
} 