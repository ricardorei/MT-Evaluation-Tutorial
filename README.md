# MT Evaluation Tutorial

Welcome to the MT Half-Marathon Evaluation tutorial! In this tutorial we will learn to use some tools that will be crucial in assessing MT quality in our future research!

These tools are:
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [COMET](https://github.com/Unbabel/COMET)
- [MT-Telescope](https://github.com/Unbabel/MT-Telescope)

When reporting the lexical-metrics such as BLEU a lot of times we get different results depending on implementations and/or tokenization applied. SacreBLEU [(Post, 2018)](https://aclanthology.org/W18-6319/) provides hassle-free computation of shareable, comparable, and reproducible lexical metrics.  In this tutorial we will use SacreBLEU to run both BLEU and chrF.

COMET [(Rei, 2020)](https://aclanthology.org/2020.emnlp-main.213/) is a new neural fine-tuned metric that achieves much better correlations with human judgements than widely used lexical-metrics. In this tutorial we will learn how to run COMET to get system scores and also how we can use COMET to get statistical significance between two systems.

Finally, since just looking at metric scores might not be enough, we will learn to use MT-Telescope [(Rei, 2021)](https://aclanthology.org/2021.acl-demo.9/) and how to look at different MT systems side-by-side.


## Tutorial Virtual Enviroment

Before we get started please create a clean virtual enviroment for this tutorial using.
If you are using python >3.8 you can simply run:

```bash
python3 -m venv /path/to/new/virtual/environment
# If the above command does not work, try this one:
# virtualenv -p python3 /path/to/new/virtual/environment
```
Then activate the venv with:

```bash
source /path/to/new/virtual/environment/bin/activate
```

## Tutorial Data:
Download the tutorial data using the following command:

```bash
wget https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/mt-half-marathon/tutorial-data.tar.gz
# Or alternatively download it from this repo.

tar -xf tutorial-data.tar.gz

cd tutorial-data
```

This is data from the [Conference on Marchine Translation (WMT20)](https://aclanthology.org/2020.wmt-1.1/) for russian-english.
We will compare the translations of two systems: OnlineA and OnlineB.

# Part I - SacreBLEU

## Installation

Within the tutorial enviroment
```bash
pip install --upgrade pip
pip install sacrebleu==2.0.0
```

After installing sacrebleu run the following command:
```bash
sacrebleu --help
```

and explore the different CLI option that sacrebleu offers you.


## Runing SacreBLEU metrics:
Running BLEU for OnlineA:
```bash
sacrebleu reference.en.txt -i OnlineA.en.txt -m bleu -l ru-en
```

### Exercises:
1) What is the BLEU score for OnlineB?
2) Replace BLEU with chrF. 

## Runing Paired Bootstrap Resampling:

```bash
sacrebleu reference.en.txt -i OnlineA.en.txt OnlineB.en.txt -m bleu -l ru-en -pbs -pbsn 25
```

### Exercises:
1) Are these results statistically significant? 
2) What about chrF? can we reject the null hypothesis using chrF?


# Part II - COMET

## Installation

Within the tutorial enviroment
```bash
pip install unbabel-comet==1.0.1
```

After installing COMET run the following command:
```bash
comet-score --help
```

## Runing COMET:

Running COMET for OnlineA:
```bash
comet-score -s source.ru.txt -t OnlineA.en.txt -r reference.en.txt --gpus 0
# If you have a GPU available you can run:
# comet-score -s source.ru.txt -t OnlineA.en.txt -r reference.en.txt --gpus 1
```

Note: COMET is much more slower than lexical metrics! If the above command is taking too long replace the model with a lightweight model using the following flag `--model wmt21-cometinho-da`.

### Exercises:

_1) What is the score of OnlineA according to COMET?_
_2) What about OnlineB?_

COMET also provides a way to look at statistical significance using t-test and bootstrap resampling. Using the `comet-compare` command compare OnlineA with OnlineB.

_3) According to comet-compare is there any systems bellow the 0.05 threshold ? Tip: Look at the win (%)._

So far we can't really tell which system is better just by looking at overall system scores. Yet, COMET also provides you segment-level scores. Look at the segment-level scores from OnlineA and OnlineB. 

_4) Which segments from OnlineA are pushing the overall score down?_

Lets analyse segment 7:

Run the following commands:
```bash
awk 'NR==7' OnlineB.en.txt
awk 'NR==7' OnlineB.en.txt
awk 'NR==7' reference.en.txt
```

_5) What is the error in the OnlineA translation?_

Take a look at segment 27. COMET assigned a score of 1.2824 to the OnlineB translation to that sentence! This high score is typically assigned to perfect translations which is not the case (meetings != games). Also, it seems that the -0.0855 score assigned to OnlineA is too severe compared to the score assigned to OnlineB.

```bash
awk 'NR==6 || NR==7 || NR==14 || NR==27' source.ru.txt > ua-source.ru.txt
awk 'NR==6 || NR==7 || NR==14 || NR==27' OnlineA.en.txt > ua-OnlineA.en.txt
awk 'NR==6 || NR==7 || NR==14 || NR==27' OnlineB.en.txt > ua-OnlineB.en.txt
awk 'NR==6 || NR==7 || NR==14 || NR==27' reference.en.txt > ua-reference.en.txt
```

_6) Run comet-score for both systems with the `--mc_dropout 15` flag. What is the recomputed score for segment 27 (segment 3) for both systems?_

Using [Uncertainty techniques](https://aclanthology.org/2021.findings-emnlp.330/) such as Monte Carlo Dropout seems to "correct" some COMET predictions yet ideally we would like the variance to give us more reliable information about samples where the model is not prroviding good estimates. Keep posted for more research in this direction and contact [Taisiya Glushkova](taisiya.glushkova@tecnico.ulisboa.pt) if you are interested in this topic!


# Part III - MT-Telescope


Looking at different translations produced by our systems is always a good idea yet, if we are using large testsets, the approach we used above becomes impossible. Fortanely, its much easier to perform these analysis using MT-Telescope!

Within the tutorial enviroment
```bash
pip install mt-telescope==0.0.2
```

MT-Telescope can be used in two different ways: with the command line or, more interactively, using its web interface. To use the web interface simply run:
```bash
telescope streamlit
```

## Comparing two MT outputs:

After lauching the web interface, you need to upload the files related to the source, reference, and translations from systems OnlineA and OnlineB (the same from the previous exercises). The language-pair also need to be inserted.

 Regarding the metrics that will be computated: by default it will be `COMET`, `chrF`, and `BLEU`, but you can choose any combination from the ones avaliable. For the tutorial, however, we recommend you to keep the default ones due to time limitations. To use again the light version of `COMET`, `wmt21-cometinho-da`, you also need to export the following env variable:

```bash
export COMET_MODEL=wmt21-cometinho-da
```

_1) Compare the outputs from the two systems._

If using the command line:

```
telescope compare -s {path/to/sources} -r {path/to/references} -x {path/to/systemX} -y {path/to/systemY} -m COMET -m BLEU -m BERTScore -l {target_language} -o {path/to/folder} --bootstrap
```
_a) Which system is the best one?_

_b) If you were to choose one of them to deploy in production, how would you choose?_

_c) Interpret the results from the bootstraping._ 

_d) If using the web inferface, try changing the thresholds for the bucket analysis plot._ 

## Dynamic Corpus Filtering:

MT-Telescope also provides functionality to dynamically evaluate sub-samples of the system outputs as a means of focused analysis tailored to a particular phenomena. 

_2) Lets start by using the Named Entities filter! For that, you can select the filter named-entities in the side bar, or in the command line by adding the flag `--filter named-entities`_.

_a) Did the overall results change?_

_b) Do you find the segment-level comparion plot useful?_

_3) Another common weakness of some MT systems is their inability to accurately translate long segments [(Koehn and Knowles, 2017)](https://aclanthology.org/W17-3204/). Having this into consideration, MT-Telescope also provides the functionality of chosing the interval of segments to be tested according to their length. In this exercise, choose the top 50% longest segments and compare again the results._



_4) [Go the extra mile!] What other features do you think are missing in MT-Telescope? Try to add them and submit a MR in the MT-Telescope repo!_ 

We hope to have convince you that a comparison between two MT systems should be more fine-grained than just looking at the corpus level metrics scores. Moreover, when possible, several metrics should be use and tests of statistical significance. 
