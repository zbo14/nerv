// #![feature(test)]
#![deny(missing_docs)]

//!Neural nets with:

//!* online learning
//!     * forward propagation of values
//!     * back propagation of errors
//!     * bias and weight updates for each neuron
//!     * repeat!
//!* variable number of hidden layers
//!* different numbers of neurons per hidden layer
//!* choice of transfer function
//!     * sigmoid 
//!     * hyperbolic tangent (tanh)
//!     * rectifier
//!     * leaky rectifier
//!
//!## Example
//!The following example is adapted from this [tutorial](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/), which uses the [wheat seeds dataset](http://archive.ics.uci.edu/ml/datasets/seeds) from the UCI Machine Learning repository. 
//!
//!### Code
//!```rust
//!extern crate csv;
//!extern crate nerv;
//!extern crate rand;
//!
//!use csv::Reader;
//!use nerv::NNet;
//!use rand::{Rng,thread_rng};
//!use std::f64::INFINITY;
//!
//!const NUM_INPUTS : usize = 7;
//!const NUM_OUTPUTS : usize = 3;
//!
//!fn main() {
//!
//!     let epochs = 500;
//!     let learn_rate = 0.3;
//!     let num_folds = 5;
//!     let layers = &[NUM_INPUTS, 5, NUM_OUTPUTS];
//!
//!     // Create a new nnet
//!     let mut rng = thread_rng();
//!     let mut nnet = NNet::new(layers, "sigmoid", &mut rng).unwrap();
//!
//!     // Read csv into Vec<Vec<f64>>
//!     let mut r = Reader::from_path("seeds.csv").unwrap();
//!     let mut data = r.deserialize::<Vec<f64>>().map(|rec| rec.unwrap()).collect::<Vec<_>>();
//!
//!     // Normalize inputs
//!     normalize_inputs(data.as_mut_slice());
//!
//!     // Shuffle data, split into train and test set
//!     rng.shuffle(&mut data);
//!     let rows_per_fold = data.len() / num_folds;
//!     let (test_data, train_data) = data.split_at_mut(rows_per_fold);
//!
//!     // Train
//!     nnet.train_online(train_data, epochs, learn_rate).unwrap();
//!
//!     // Predict and check results
//!     check_accuracy(&mut nnet, test_data);
//!}
//!
//!fn normalize_inputs(data: &mut[Vec<f64>]) {
//!     let mut min;
//!     let mut max;
//!     for i in 0..NUM_INPUTS {
//!         min = INFINITY;
//!         max = -INFINITY;
//!         for row in data.iter() {
//!             if row[i] > max {
//!                 max = row[i];
//!             } else if row[i] < min {
//!                 min = row[i]
//!             }
//!         }
//!         let diff = max - min;
//!         for row in data.iter_mut() {
//!             row[i] = (row[i] - min) / diff;
//!         }
//!     }
//!}
//!
//!fn check_accuracy(nnet: &mut NNet, data: &mut [Vec<f64>]) {
//!     let mut num_correct = 0;
//!     for row in data.iter_mut() {
//!         let (inputs, outputs) = row.split_at_mut(NUM_INPUTS);
//!         let predicts = nnet.predict(inputs).unwrap();
//!         let i = predicts.iter().enumerate().max_by(|&(_,v1), &(_,v2)| {
//!             v1.partial_cmp(&v2).unwrap()
//!         }).unwrap().0;
//!         if outputs[i] == 1.0 {
//!             num_correct += 1;
//!         }
//!     } 
//!     println!("Accuracy: {} / {}", num_correct, data.len());
//!}
//!```

extern crate rand;
// extern crate test;

use self::rand::{Rng,ThreadRng};

fn rand(rng: &mut ThreadRng) -> f64 {
    rng.gen::<f64>()
}

fn rand_clamped(rng: &mut ThreadRng) -> f64 {
    rng.gen::<f64>() - rng.gen::<f64>()
}

struct Transfer{
    f: fn(f64) -> f64,
    df: fn(f64) -> f64,
    rand: fn(&mut ThreadRng) -> f64,
}

fn f_rectifier(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn df_rectifier(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn rectifier_transfer() -> Transfer {
    Transfer{
        f: f_rectifier,
        df: df_rectifier,
        rand: rand_clamped,
    }
}

fn f_leaky_rectifier(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

fn df_leaky_rectifier(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.01
    }
}

fn leaky_rectifier_transfer() -> Transfer {
    Transfer{
        f: f_leaky_rectifier,
        df: df_leaky_rectifier,
        rand: rand_clamped,
    }
}

fn f_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn df_sigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}

fn sigmoid_transfer() -> Transfer {
    Transfer{
        f: f_sigmoid, 
        df: df_sigmoid,
        rand,
    }
}

fn f_tanh(x: f64) -> f64 {
    let e1 = x.exp();
    let e2 = (-x).exp();
    (e1 - e2) / (e1 + e2)
}

fn df_tanh(x: f64) -> f64 {
    1.0 / (x.exp() + (-x).exp()).powi(2)
}

fn tanh_transfer() -> Transfer {
    Transfer{
        f: f_tanh, 
        df: df_tanh,
        rand: rand_clamped,
    }
}

struct Neuron {
    bias: f64,
    delta: f64,
    value: f64,
    weights: Vec<f64>,
}

impl Neuron {

    fn new(num_inputs: usize, transfer: &Transfer, rng: &mut ThreadRng) -> Neuron {
        let bias = (transfer.rand)(rng);
        let weights = (0..num_inputs).map(|_| (transfer.rand)(rng)).collect::<Vec<_>>();
        Neuron{
            bias,
            delta: 0.0,
            value: 0.0,
            weights,
        }
    }

    fn backward(&mut self, error: f64, transfer: &Transfer) {
        self.delta += error * (transfer.df)(self.value);
    }

    fn forward_inputs(&mut self, inputs: &[f64], transfer: &Transfer) {
        self.value = (transfer.f)(self.weights.iter().zip(inputs.iter()).fold(self.bias, |acc, (w, x)| acc + w * x));
    }

    fn forward_layer(&mut self, layer: &[Neuron], transfer: &Transfer) {
        self.value = (transfer.f)(self.weights.iter().zip(layer.iter()).fold(self.bias, |acc, (w, neuron)| acc + w * neuron.value));
    }

    fn update_inputs(&mut self, inputs: &[f64], learn_rate: f64) {
        for (i, x) in inputs.iter().enumerate() {
            self.weights[i] += learn_rate * self.delta * x;
        }
        self.bias += learn_rate * self.delta;
        self.delta = 0.0;
    }

    fn update_layer(&mut self, layer: &[Neuron], learn_rate: f64) {
        for (i, neuron) in layer.iter().enumerate() {
            self.weights[i] += learn_rate * self.delta * neuron.value;
        }
        self.bias += learn_rate * self.delta;
        self.delta = 0.0;
    }
}

/// The neural network
pub struct NNet {
    error: f64,
    layers: Vec<Vec<Neuron>>,
    num_hidden_layers: usize,
    num_layers: usize,
    num_inputs: usize,
    num_outputs: usize,
    outputs: Vec<f64>,
    row_size: usize,
    transfer: Transfer,
}

impl NNet {

    ///Create a new nnet
    ///# Example
    ///Here's a nnet with:
    ///
    /// * 4 inputs
    /// * 3 hidden layers with 5, 6, and 7 neurons respectively
    /// * 2 outputs
    /// * sigmoid transfer
    /// 
    ///```
    ///extern crate nerv; 
    ///extern crate rand;
    ///
    ///use nerv::NNet;
    ///use rand::thread_rng;
    ///
    ///fn main() {
    ///    let layers = &[4, 5, 6, 7, 2];
    ///    let mut rng = thread_rng();
    ///    let mut nnet = NNet::new(layers, "sigmoid", &mut rng);
    ///    // ...
    ///}
    ///```
    pub fn new(layers: &[usize], transfer: &str, rng: &mut ThreadRng) -> Result<NNet,String> {
        let transfer = match transfer {
            "leaky_rectifier" => Ok(leaky_rectifier_transfer()),
            "rectifier" => Ok(rectifier_transfer()),
            "sigmoid" => Ok(sigmoid_transfer()),
            "tanh" => Ok(tanh_transfer()),
            _ => Err(format!("unexpected transfer: {}", transfer)),
        }?;
        let num_layers = layers.len() - 1;
        let num_hidden_layers = num_layers - 1;
        let num_inputs = layers[0];
        let num_outputs = layers[num_layers];
        let outputs = vec![0.0; num_outputs];
        let row_size = num_inputs + num_outputs;
        let layers = (1..layers.len()).map(|i| {
            NNet::layer(layers[i-1], layers[i], &transfer, rng)
        }).collect::<Vec<_>>();
        let nnet = NNet{
            error: 0.0,
            layers,
            num_inputs,
            num_outputs,
            num_hidden_layers,
            num_layers,
            outputs,
            row_size,
            transfer,
        };
        Ok(nnet)
    }

    /// Train nnet with online learning
    pub fn train_online(&mut self, train_data: &[Vec<f64>], epochs: usize, learn_rate: f64) -> Result<(),String> {
        println!("Online training - errors");
        println!("learn_rate={}", learn_rate);
        for epoch in 0..epochs {
            self.error = 0.0;
            for row in train_data.iter() {
                let row_size = row.len();
                if row_size != self.row_size {
                    return Err(format!("expected {} data in row, got {}", self.row_size, row_size))
                }
                let (inputs, outputs) = row.split_at(self.num_inputs);
                self.forward(inputs);
                self.backward(outputs);
                self.update(inputs, learn_rate);
            }
            println!("[{}] {}", epoch, self.error);
        }
        Ok(())
    }

    fn train_batch(&mut self, train_data: &[Vec<f64>], epochs: usize, learn_rate: f64) {
        println!("Batch training - errors");
        println!("learn_ rate={}", learn_rate);
        for epoch in 0..epochs {
            self.error = 0.0;
            for row in train_data.iter() {
                let (inputs, outputs) = row.split_at(self.num_inputs);
                self.forward(inputs);
                self.backward(outputs);
            }
            for row in train_data.iter() {
                let inputs = &row[..self.num_inputs];
                self.update(inputs, learn_rate);
            }
            println!("[{}] {}", epoch, self.error);
        }
    }

    /// Predict outputs for input values
    pub fn predict(&mut self, inputs: &[f64]) -> Result<&[f64],String> {
        let num_inputs = inputs.len();
        if num_inputs != self.num_inputs {
            return Err(format!("expected {} inputs, got {}", self.num_inputs, num_inputs))
        }
        self.forward(inputs);
        for (i, neuron) in self.layers[self.num_hidden_layers].iter().enumerate() {
            self.outputs[i] = neuron.value;
        }
        Ok(&self.outputs)
    }

    fn backward(&mut self, outputs: &[f64]) {
        for (neuron, output) in self.layers[self.num_hidden_layers].iter_mut().zip(outputs) {
            let error = output - neuron.value;
            neuron.backward(error, &self.transfer);
            self.error += error.abs();
        }
        for i in (2..self.num_layers+1).rev() {
            let (layer, layers) = self.layers[..i].split_last_mut().unwrap();
            for (j, neuron) in layers[i-2].iter_mut().enumerate() {
                let error = layer.iter().fold(0.0, |acc, n| acc + n.weights[j] * n.delta);
                neuron.backward(error, &self.transfer);
            }
        }
    }

    fn forward(&mut self, inputs: &[f64]) {
        for neuron in self.layers[0].iter_mut() {
            neuron.forward_inputs(inputs, &self.transfer);
        }
        for i in 0..self.num_hidden_layers {
            let (layer, layers) = self.layers[i..].split_first_mut().unwrap();
            for neuron in layers[0].iter_mut() {
                neuron.forward_layer(layer, &self.transfer);
            }
        }
    }

    fn update(&mut self, inputs: &[f64], learn_rate: f64) {
        for i in (2..self.num_layers+1).rev() {
            let (layer, layers) = self.layers[..i].split_last_mut().unwrap();
            for neuron in layer.iter_mut() {
                neuron.update_layer(&layers[i-2], learn_rate);
            }
        }
        for neuron in self.layers[0].iter_mut() {
            neuron.update_inputs(inputs, learn_rate);
        }
    }

    fn layer(num_inputs: usize, num_neurons: usize, transfer: &Transfer, rng: &mut ThreadRng) -> Vec<Neuron> {
        (0..num_neurons).map(|_| Neuron::new(num_inputs, transfer, rng)).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {

    extern crate csv;

    use super::*;
    use std::f64::INFINITY;
    // use self::test::Bencher;

    // single hidden layer test case and wheat seeds example adapted from: 
    // https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

    fn round_decimal_places(x: f64, dec_places: i32) -> f64 {
        let pow10 = (10 as f64).powi(dec_places);
        (x * pow10).round() / pow10
    }

    fn neuron(bias: f64, weights: Vec<f64>) -> Neuron {
        Neuron{
            bias,
            delta: 0.0,
            value: 0.0,
            weights,
        }
    }

    fn single_hidden_layer() -> Vec<Vec<Neuron>> {
        vec![
            vec![
            neuron(0.763774618976614, vec![0.13436424411240122, 0.8474337369372327])],
            vec![
            neuron(0.49543508709194095, vec![0.2550690257394217]),
            neuron(0.651592972722763, vec![0.4494910647887381])]
        ]
    }

    fn nnet_with_single_hidden_layer() -> NNet {
        NNet{
            error: 0.0,
            num_inputs: 2,
            num_outputs: 2,
            num_hidden_layers: 1,
            num_layers: 2,
            layers: single_hidden_layer(),
            outputs: Vec::new(),
            row_size: 4,
            transfer: sigmoid_transfer(),
        }
    }

    fn check_single_hidden_layer_deltas(nnet: &mut NNet) {
        let hidden_delta = round_decimal_places(nnet.layers[0][0].delta, 4);
        let output1_delta = round_decimal_places(nnet.layers[1][0].delta, 4);
        let output2_delta = round_decimal_places(nnet.layers[1][1].delta, 4);
        assert_eq!(hidden_delta, -0.0005);
        assert_eq!(output1_delta, -0.1462);
        assert_eq!(output2_delta, 0.0772);
    }

    fn check_single_hidden_layer_weights(nnet: &mut NNet) {
        let neuron1 = &nnet.layers[0][0];
        let neuron2 = &nnet.layers[1][0];
        let neuron3 = &nnet.layers[1][1];
        let weight1 = round_decimal_places(neuron1.weights[0], 4);
        let weight2 = round_decimal_places(neuron1.weights[1], 4);
        let weight3 = round_decimal_places(neuron2.weights[0], 4);
        let weight4 = round_decimal_places(neuron3.weights[0], 4);
        assert_eq!(weight1, 0.1341); // 0.13409684171007069415
        assert_eq!(weight2, 0.8474); // 
        assert_eq!(weight3, 0.2031); // 0.203129909228224925631078630828836
        assert_eq!(weight4, 0.4769); // 0.47690913283740555
        let bias1 = round_decimal_places(neuron1.bias, 4);
        let bias2 = round_decimal_places(neuron2.bias, 4);
        let bias3 = round_decimal_places(neuron3.bias, 4);
        assert_eq!(bias1, 0.7635); // 0.76350721657428347415
        assert_eq!(bias2, 0.4223); // 0.42233976367402691
        assert_eq!(bias3, 0.6902); // 0.6901791614400793
    }

    fn reset_single_hidden_layer(nnet: &mut NNet) {
        {
            let neuron1 = &mut nnet.layers[0][0];
            neuron1.bias = 0.763774618976614;
            neuron1.delta = 0.0;
            neuron1.value = 0.7105668883115941;
            neuron1.weights = vec![0.13436424411240122, 0.8474337369372327];
        }
        {   
            let neuron2 = &mut nnet.layers[1][0];
            neuron2.bias = 0.49543508709194095;
            neuron2.delta = 0.0;
            neuron2.value = 0.6213859615555266;
            neuron2.weights = vec![0.2550690257394217];
        }
        {
            let neuron3 = &mut nnet.layers[1][1];
            neuron3.bias = 0.651592972722763;
            neuron3.delta = 0.0;
            neuron3.value = 0.6573693455986976;
            neuron3.weights = vec![0.4494910647887381];
        }
    }

    fn forward_single_hidden_layer(nnet: &mut NNet) {
        nnet.forward(&[1.0, 0.0]);
        let output_layer = &nnet.layers[nnet.num_hidden_layers];
        let output1 = round_decimal_places(output_layer[0].value, 4);
        let output2 = round_decimal_places(output_layer[1].value, 4);
        assert_eq!(output1, 0.6630);
        assert_eq!(output2, 0.7253);
    }

    fn backward_single_hidden_layer(nnet: &mut NNet) {
        reset_single_hidden_layer(nnet);
        nnet.backward(&[0.0, 1.0]);
        check_single_hidden_layer_deltas(nnet);
    }

    fn update_single_hidden_layer(nnet: &mut NNet) {
        nnet.update(&[1.0, 0.0], 0.5);
        check_single_hidden_layer_weights(nnet);
    }

    #[test]
    fn test_single_hidden_layer() {
        let mut nnet = nnet_with_single_hidden_layer();
        forward_single_hidden_layer(&mut nnet);
        backward_single_hidden_layer(&mut nnet);
        update_single_hidden_layer(&mut nnet);
    }

    fn multiple_hidden_layers() -> Vec<Vec<Neuron>> {
        vec![
            vec![                                           // > FORWARD            // > BACKWARD               // > UPDATE
            neuron(0.01, vec![0.02, 0.03]),                 // 0.5087491068802403   // 0.0001056202957707497    // 0.010052810147885375     [0.020026405073942688, 0.030026405073942688]
            neuron(0.04, vec![0.05, 0.06]),                 // 0.5237321541265610   // 0.0001115973965475908    // 0.040055798698273797     [0.0500278993491369, 0.060027899349136896]
            neuron(0.07, vec![0.08, 0.09])],                // 0.5386726052065080   // 0.0001173399248409821    // 0.07005866996242         [0.08002933498121, 0.09002933498121]
            vec![
            neuron(0.10, vec![0.11, 0.12, 0.13]),           // 0.5717115586037880   // 0.0006085746924161419    // 0.10030428734620808      [0.11015480591561831, 0.12015936506730301, 0.13016391125751328]
            neuron(0.14, vec![0.15, 0.16, 0.17]),           // 0.5966879869619279   // 0.0006184439246096836    // 0.1403092219623          [0.15015731639715033, 0.16016194948442117, 0.17016656940002184]
            neuron(0.18, vec![0.19, 0.20, 0.21]),           // 0.6211730043885718   // 0.0006246078584674122    // 0.1803123039292337       [0.19015888434507283, 0.2001635636096, 0.21016822957117653]
            neuron(0.22, vec![0.23, 0.24, 0.25])],          // 0.6450557844800172   // 0.0006270664486428652    // 0.22031353322432143      [0.23015950974785082, 0.2401642074309641, 0.250168891758764]
            vec![
            neuron(0.26, vec![0.27, 0.28, 0.29, 0.30]),     // 0.7221379989948211   // 0.0043394090620597825    // 0.26216970453102989125   [0.2712404451591448, 0.2812946366289224, 0.2913477618821753, 0.3013995804583533]
            neuron(0.31, vec![0.32, 0.33, 0.34, 0.35])],    // 0.7552521334389936   // 0.0041055749278109330    // 0.3120527874639054665    [0.3211736023204717, 0.3312248736195, 0.3412751361563254, 0.3513241624279]
            vec![
            neuron(0.36, vec![0.37, 0.38])]                 // 0.7138561612920588   // 0.0584493263830216400    // 0.38922466319151082      [0.3911042397984, 0.4020719892244]
        ]
    }

    fn nnet_with_multiple_hidden_layers() -> NNet {
        NNet{
            error: 0.0,
            num_inputs: 2,
            num_outputs: 1,
            num_hidden_layers: 3,
            num_layers: 4,
            layers: multiple_hidden_layers(),
            outputs: Vec::new(),
            row_size: 3,
            transfer: sigmoid_transfer(),
        }
    }

    fn forward_multiple_hidden_layers(nnet: &mut NNet) {
        nnet.forward(&[0.5, 0.5]);
        let value1 = round_decimal_places(nnet.layers[0][0].value, 4);
        let value2 = round_decimal_places(nnet.layers[0][1].value, 4);
        let value3 = round_decimal_places(nnet.layers[0][2].value, 4);
        assert_eq!(value1, 0.5087);
        assert_eq!(value2, 0.5237);
        assert_eq!(value3, 0.5387);
        let value4 = round_decimal_places(nnet.layers[1][0].value, 4);
        let value5 = round_decimal_places(nnet.layers[1][1].value, 4);
        let value6 = round_decimal_places(nnet.layers[1][2].value, 4);
        let value7 = round_decimal_places(nnet.layers[1][3].value, 4);
        assert_eq!(value4, 0.5717);
        assert_eq!(value5, 0.5967);
        assert_eq!(value6, 0.6212);
        assert_eq!(value7, 0.6451);
        let value8 = round_decimal_places(nnet.layers[2][0].value, 4);
        let value9 = round_decimal_places(nnet.layers[2][1].value, 4);
        assert_eq!(value8, 0.7221);
        assert_eq!(value9, 0.7553);
        let value10 = round_decimal_places(nnet.layers[3][0].value, 4);
        assert_eq!(value10, 0.7139);
    }

    fn check_multiple_hidden_layer_deltas(nnet: &mut NNet) {
        let delta1 = round_decimal_places(nnet.layers[3][0].delta, 4);
        assert_eq!(delta1, 0.0584);
        let delta2 = round_decimal_places(nnet.layers[2][0].delta, 4);
        let delta3 = round_decimal_places(nnet.layers[2][1].delta, 4);
        assert_eq!(delta2, 0.0043);
        assert_eq!(delta3, 0.0041);
        let delta4 = round_decimal_places(nnet.layers[1][0].delta, 4);
        let delta5 = round_decimal_places(nnet.layers[1][1].delta, 4);
        let delta6 = round_decimal_places(nnet.layers[1][2].delta, 4);
        let delta7 = round_decimal_places(nnet.layers[1][3].delta, 4);
        assert_eq!(delta4, 0.0006);
        assert_eq!(delta5, 0.0006);
        assert_eq!(delta6, 0.0006);
        assert_eq!(delta7, 0.0006);
        let delta8 = round_decimal_places(nnet.layers[0][0].delta, 4);
        let delta9 = round_decimal_places(nnet.layers[0][1].delta, 4);
        let delta10 = round_decimal_places(nnet.layers[0][2].delta, 4);
        assert_eq!(delta8, 0.0001);
        assert_eq!(delta9, 0.0001);
        assert_eq!(delta10, 0.0001);
    }

    fn check_multiple_hidden_layer_weights(nnet: &mut NNet) {
        let biases = vec![
            0.0101, 0.0401, 0.0701,
            0.1003, 0.1403, 0.1803, 0.2203,
            0.2622, 0.3121, 
            0.3892];
        let weights = vec![
            0.0200, 0.0300,
            0.0500, 0.0600,
            0.0800, 0.0900,
            0.1102, 0.1202, 0.1302,
            0.1502, 0.1602, 0.1702,
            0.1902, 0.2002, 0.2102,
            0.2302, 0.2402, 0.2502,
            0.2712, 0.2813, 0.2913, 0.3014,
            0.3212, 0.3312, 0.3413, 0.3513,
            0.3911, 0.4021];
        let mut i = 0;
        let mut j = 0;
        for layer in nnet.layers.iter() {
            for neuron in layer.iter() {
                let bias = round_decimal_places(neuron.bias, 4);
                assert_eq!(bias, biases[i]);
                i += 1;
                for &weight in neuron.weights.iter() {
                    let weight = round_decimal_places(weight, 4);
                    assert_eq!(weight, weights[j]);
                    j += 1;
                }
            }
        }
    }

    fn backward_multiple_hidden_layers(nnet: &mut NNet) {
        nnet.backward(&[1.0]);
        check_multiple_hidden_layer_deltas(nnet);
    }

    fn update_multiple_hidden_layers(nnet: &mut NNet) {
        nnet.update(&[0.5, 0.5], 0.5);
        check_multiple_hidden_layer_weights(nnet);
    }

    #[test]
    fn test_multiple_hidden_layers() {
        let mut nnet = nnet_with_multiple_hidden_layers();
        forward_multiple_hidden_layers(&mut nnet);
        backward_multiple_hidden_layers(&mut nnet);
        update_multiple_hidden_layers(&mut nnet);
    }

    fn data() -> Vec<Vec<f64>> {
        vec![
            vec![2.7810836, 2.550537003, 1.0, 0.0],
            vec![1.465489372, 2.362125076, 1.0, 0.0],
            vec![3.396561688, 4.400293529, 1.0, 0.0],
            vec![1.38807019, 1.850220317, 1.0, 0.0],
            vec![3.06407232, 3.005305973, 1.0, 0.0],
            vec![7.627531214, 2.759262235, 0.0, 1.0],
            vec![5.332441248, 2.088626775, 0.0, 1.0],
            vec![6.922596716, 1.77106367, 0.0, 1.0],
            vec![8.675418651, -0.242068655, 0.0, 1.0], 
            vec![7.673756466, 3.508563011, 0.0, 1.0]
        ]
    }

    #[test]
    #[should_panic]
    fn test_unexpected_transfer() {
        let mut rng = rand::thread_rng();
        NNet::new(&[2, 2, 2], "softmax", &mut rng).unwrap();
    }

    #[test]
    #[should_panic]
    fn train_with_wrong_num_inputs() {
        let mut nnet = nnet_with_single_hidden_layer();
        let mut data = data();
        data[1].push(0.1);
        nnet.train_online(&data, 20, 0.5).unwrap(); 
    }

    #[test]
    #[should_panic]
    fn predict_with_wrong_num_inputs() {
        let mut nnet = nnet_with_single_hidden_layer();
        nnet.predict(&[0.1, 0.2, 0.3]).unwrap();
    }

    #[test]
    #[ignore]
    fn train_batch() {
        let data = data();
        let mut rng = rand::thread_rng();
        let mut nnet = NNet::new(&[2, 2, 2], "sigmoid", &mut rng).unwrap();
        nnet.train_batch(&data, 20, 0.5);
    }

    #[test]
    fn train_online() {
        let data = data();
        let mut rng = rand::thread_rng();
        let mut nnet = NNet::new(&[2, 2, 2], "sigmoid", &mut rng).unwrap();
        nnet.train_online(&data, 20, 0.5).unwrap();
    }

    const NUM_INPUTS : usize = 7;
    const NUM_OUTPUTS : usize = 3;

    fn check_accuracy(nnet: &mut NNet, data: &mut [Vec<f64>]) {
        let mut num_correct = 0;
        for row in data.iter_mut() {
            let (inputs, outputs) = row.split_at_mut(NUM_INPUTS);
            let predicts = nnet.predict(inputs).unwrap();
            let i = predicts.iter().enumerate().max_by(|&(_,v1), &(_,v2)| {
                v1.partial_cmp(&v2).unwrap()
            }).unwrap().0;
            if outputs[i] == 1.0 {
                num_correct += 1;
            }
        } 
        println!("Accuracy: {} / {}", num_correct, data.len());
    }

    fn normalize_inputs(data: &mut[Vec<f64>]) {
        let mut min;
        let mut max;
        for i in 0..NUM_INPUTS {
            min = INFINITY;
            max = -INFINITY;
            for row in data.iter() {
                if row[i] > max {
                    max = row[i];
                } else if row[i] < min {
                    min = row[i]
                }
            }
            let diff = max - min;
            for row in data.iter_mut() {
                row[i] = (row[i] - min) / diff;
            }
        }
    }

    #[test]
    fn test_example() {

        let epochs = 500;
        let learn_rate = 0.3;
        let num_folds = 5;
        let layers = &[NUM_INPUTS, 5, NUM_OUTPUTS];

        // Create a new nnet
        let mut rng = rand::thread_rng();
        let mut nnet = NNet::new(layers, "sigmoid", &mut rng).unwrap();

        // Read csv into Vec<Vec<f64>>
        let mut r = csv::Reader::from_path("seeds.csv").unwrap();
        let mut data = r.deserialize::<Vec<f64>>().map(|rec| rec.unwrap()).collect::<Vec<_>>();

        // Normalize inputs
        normalize_inputs(data.as_mut_slice());

        // Shuffle data, split into train and test set
        rng.shuffle(&mut data);
        let rows_per_fold = data.len() / num_folds;
        let (test_data, train_data) = data.split_at_mut(rows_per_fold);

        // Train
        nnet.train_online(train_data, epochs, learn_rate).unwrap();

        // Predict and check results
        check_accuracy(&mut nnet, test_data);
    }

    /*

    #[bench]
    fn epoch_with_single_hidden_layer(b: &mut Bencher) {
        let mut nnet = nnet_with_single_hidden_layer();
        let inputs = &[0.5, 0.5];
        let outputs = &[0.7, 0.3];
        let learn_rate = 0.3;
        b.iter(|| {
            nnet.forward(inputs);
            nnet.backward(inputs);
            nnet.update(outputs, learn_rate);
        });
    }

    #[bench]
    fn epoch_with_multiple_hidden_layers(b: &mut Bencher) {
        let mut nnet = nnet_with_multiple_hidden_layers();
        let inputs = &[0.4, 0.6];
        let outputs = &[0.8];
        let learn_rate = 0.2;
        b.iter(|| {
            nnet.forward(inputs);
            nnet.backward(outputs);
            nnet.update(inputs, learn_rate);
        });
    }

    */
}