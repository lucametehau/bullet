/*
This is about as simple as you can get with a network, the arch is
    (768 -> HIDDEN_SIZE)x2 -> 1
and the training schedule is pretty sensible.
There's potentially a lot of elo available by adjusting the wdl
and lr schedulers, depending on your dataset.
*/
use bullet_lib::{
    game::{
        formats::sfbinpack::{
            chess::{r#move::MoveType, piecetype::PieceType},
            TrainingDataEntry,
        },
        inputs::{self, get_num_buckets},
        outputs::MaterialCount
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

use viriformat::dataformat::Filter;

const HIDDEN_SIZE: usize = 1280;
const SCALE: i32 = 225;
const QA: i16 = 255;
const QB: i16 = 64;
const NUM_OUTPUT_BUCKETS: usize = 8;

const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    0, 1, 2, 3,
    4, 4, 5, 5,
    4, 4, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    6, 6, 6, 6,
    6, 6, 6, 6,
];

const STAGE1_SB: usize = 800;
const STAGE2_SB: usize = 100;

fn main() {
    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let mut trainer = ValueTrainerBuilder::default()
        // makes `ntm_inputs` available below
        .dual_perspective()
        // standard optimiser used in NNUE
        // the default AdamW params include clipping to range [-1.98, 1.98]
        .optimiser(AdamW)
        // basic piece-square chessboard inputs
        .inputs(inputs::ChessBucketsMirrored::new(BUCKET_LAYOUT))
        // output buckets
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        // chosen such that inference may be efficiently implemented in-engine
        .save_format(&[
            SavedFormat::id("l0w")
                .add_transform(|builder, _, mut weights| {
                    let factoriser = builder.get_weights("l0f").get_dense_vals().unwrap();
                    let expanded = factoriser.repeat(NUM_INPUT_BUCKETS);

                    for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                        *i += j;
                    }

                    weights
                })
                .quantise::<i16>(255),
            SavedFormat::id("l0b").quantise::<i16>(255),
            SavedFormat::id("l1w").quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").quantise::<i16>(255 * 64),
        ])
        // map output into ranges [0, 1] to fit against our labels which
        // are in the same range
        // `target` == wdl * game_result + (1 - wdl) * sigmoid(search score in centipawns / SCALE)
        // where `wdl` is determined by `wdl_scheduler`
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        // the basic `(768 -> N)x2 -> 1` inference
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // weights
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // output layer weights
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: "net1280-interleaved11-28".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: STAGE1_SB + STAGE2_SB,
        },
        wdl_scheduler: wdl::Sequence { 
            first: wdl::ConstantWDL { value: 0.4 },
            second: wdl::ConstantWDL { value: 0.5 },
            first_scheduler_final_superbatch: STAGE1_SB,
        },
        lr_scheduler: lr::Sequence { 
            first: lr::CosineDecayLR { 
                initial_lr: 0.001, 
                final_lr: 0.001 * 0.3 * 0.3 * 0.3, 
                final_superbatch: STAGE1_SB 
            },
            second: lr::CosineDecayLR { 
                initial_lr: 0.001 * 0.3 * 0.3 * 0.3, 
                final_lr: 0.001 * 0.3 * 0.3 * 0.3 * 0.1, 
                final_superbatch: STAGE2_SB 
            },
            first_scheduler_final_superbatch: STAGE1_SB,
        },
        save_rate: 200,
    };
    
    // retrain schedule
    // let schedule = TrainingSchedule {
    //     net_id: "net1280-interleaved11-26-factorised-wdl04-retrain".to_string(),
    //     eval_scale: SCALE as f32,
    //     steps: TrainingSteps {
    //         batch_size: 16_384,
    //         batches_per_superbatch: 6104,
    //         start_superbatch: 1,
    //         end_superbatch: 100,
    //     },
    //     wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
    //     lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001 * 0.3 * 0.3 * 0.3, final_lr: 0.001 * 0.3 * 0.3 * 0.3 * 0.1, final_superbatch: 100 },
    //     save_rate: 100,
    // };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let filter = Filter {
        min_ply: 16,
        min_pieces: 4,
        max_eval: 8000,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,
        random_fen_skipping: false,
        random_fen_skip_probability: 0.0,
        wdl_filtered: false,
        wdl_model_params_a: [6.871_558_62, -39.652_263_91, 90.684_603_52, 170.669_963_64],
        wdl_model_params_b: [
            -7.198_907_10,
            56.139_471_85,
            -139.910_911_83,
            182.810_074_27,
        ],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.5,
    };
    // loading directly from a `BulletFormat` file
    let data_loader = loader::ViriBinpackLoader::new("/root/interleaved11-28.bin", 2048, 4, filter);

    // let data_loader = DirectSequentialDataLoader::new(&["G://archive//run_2024-01-03_22-34-48_5000000g-64t-no_tb-nnue-dfrc-n5000-bf.bin"]);
    // let data_loader = DirectSequentialDataLoader::new(&["G://CloverData//Clover-20k-bf-shuffled.bin"]);
    // trainer.load_from_checkpoint("/root/bullet/checkpoints/net1280-interleaved-11-26-factorised-wdl03-04-600/");
    trainer.run(&schedule, &settings, &data_loader);
}

/*
This is how you would load the network in rust.
Commented out because it will error if it can't find the file.
static NNUE: Network =
    unsafe { std::mem::transmute(*include_bytes!("../checkpoints/simple-10/simple-10.bin")) };
*/

#[inline]
/// Clipped ReLU - Activation Function.
/// Note that this takes the i16s in the accumulator to i32s.
fn crelu(x: i16) -> i32 {
    i32::from(x).clamp(0, i32::from(QA))
}

/// This is the quantised format that bullet outputs.
#[repr(C)]
pub struct Network {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    feature_weights: [Accumulator; 768],
    /// Vector with dimension `HIDDEN_SIZE`.
    feature_bias: Accumulator,
    /// Column-Major `1 x (2 * HIDDEN_SIZE)`
    /// matrix, we use it like this to make the
    /// code nicer in `Network::evaluate`.
    output_weights: [i16; 2 * HIDDEN_SIZE],
    /// Scalar output bias.
    output_bias: i16,
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        // Initialise output with bias.
        let mut output = i32::from(self.output_bias);

        // Side-To-Move Accumulator -> Output.
        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[..HIDDEN_SIZE]) {
            output += crelu(input) * i32::from(weight);
        }

        // Not-Side-To-Move Accumulator -> Output.
        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[HIDDEN_SIZE..]) {
            output += crelu(input) * i32::from(weight);
        }

        // Apply eval scale.
        output *= SCALE;

        // Remove quantisation.
        output /= i32::from(QA) * i32::from(QB);

        output
    }
}

/// A column of the feature-weights matrix.
/// Note the `align(64)`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; HIDDEN_SIZE],
}

impl Accumulator {
    /// Initialised with bias so we can just efficiently
    /// operate on it afterwards.
    pub fn new(net: &Network) -> Self {
        net.feature_bias
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i += *d
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i -= *d
        }
    }
}
