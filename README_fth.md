<div align="center">

Megatron-LM & Megatron-Core
===========================
<h4>GPU optimized techniques for training transformer models at-scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.5.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-OpenBSD-blue)](./LICENSE)

<div align="left">

# Latest News

- **[2024/7]** Megatron-Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/)). 
- **[2024/6]** Megatron-Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/1 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron-Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron-Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs. Explore the [Megatron-Core intro](#megatron-core) for more details.
最新动态
[2024年7月] Megatron-Core v0.7 提升了可扩展性和训练弹性，并增加了对多模态训练的支持（博客）。
[2024年6月] Megatron-Core 增加了对基于Mamba模型的支持。查看我们的论文 An Empirical Study of Mamba-based Language Models 和 代码示例。
[2024年1月公告] NVIDIA 已将Megatron-LM的核心功能发布到此存储库中的 Megatron-Core。Megatron-Core 在 Megatron-LM 的 GPU 优化技术基础上，增加了更多系统级优化的前沿创新，具有可组合和模块化的API。探索 Megatron-Core 简介 了解更多详情。

# Table of Contents

- [Megatron-LM \& Megatron-Core](#megatron-lm--megatron-core)
- [Latest News](#latest-news)
- [Table of Contents](#table-of-contents)
- [Megatron Overview](#megatron-overview)
  - [Megatron-LM](#megatron-lm)
  - [Megatron-Core](#megatron-core)
- [Training Speed and Scalability](#training-speed-and-scalability)
- [Setup](#setup)
  - [Downloading Checkpoints](#downloading-checkpoints)
- [Usage](#usage)
- [Training](#training)
  - [Data Preprocessing](#data-preprocessing)
  - [BERT Pretraining](#bert-pretraining)
  - [GPT Pretraining](#gpt-pretraining)
  - [T5 Pretraining](#t5-pretraining)
  - [Distributed Pretraining](#distributed-pretraining)
  - [Activation Checkpointing and Recomputation](#activation-checkpointing-and-recomputation)
  - [Distributed Optimizer](#distributed-optimizer)
  - [FlashAttention](#flashattention)
  - [GPT-3 Example](#gpt-3-example)
  - [Retro and InstructRetro](#retro-and-instructretro)
  - [Mamba-based Language Models](#mamba-based-language-models)
  - [Mixture of Experts](#mixture-of-experts)
- [Evaluation and Tasks](#evaluation-and-tasks)
  - [GPT Text Generation](#gpt-text-generation)
    - [Detoxify GPT via Self-generation](#detoxify-gpt-via-self-generation)
  - [GPT Evaluation](#gpt-evaluation)
    - [WikiText Perplexity Evaluation](#wikitext-perplexity-evaluation)
    - [LAMBADA Cloze Accuracy](#lambada-cloze-accuracy)
  - [BERT Task Evaluation](#bert-task-evaluation)
    - [RACE Evaluation](#race-evaluation)
    - [MNLI Evaluation](#mnli-evaluation)
  - [Llama-2 Inference and Finetuning](#llama-2-inference-and-finetuning)
- [Model Optimization and Deployment](#model-optimization-and-deployment)
  - [Quantization and TensorRT-LLM Deployment](#quantization-and-tensorrt-llm-deployment)
- [Datasets](#datasets)
  - [Collecting Wikipedia Training Data](#collecting-wikipedia-training-data)
  - [Collecting GPT Webtext Data](#collecting-gpt-webtext-data)
- [Reproducibility](#reproducibility)
- [Checkpoint conversion](#checkpoint-conversion)
  - [Model class conversion](#model-class-conversion)
  - [Checkpoint format conversion](#checkpoint-format-conversion)
- [Projects Using Megatron](#projects-using-megatron)

目录
Megatron-LM 和 Megatron-Core
最新动态
目录
Megatron 概述
Megatron-LM
Megatron-Core
训练速度与可扩展性
设置
下载检查点
使用方法
训练
数据预处理
BERT 预训练
GPT 预训练
T5 预训练
分布式预训练
激活检查点与重计算
分布式优化器
FlashAttention
GPT-3 示例
Retro 和 InstructRetro
基于Mamba的语言模型
混合专家模型
评估与任务
GPT 文本生成
通过自我生成去毒化 GPT
GPT 评估
WikiText 困惑度评估
LAMBADA 完形填空准确率
BERT 任务评估
RACE 评估
MNLI 评估
Llama-2 推理与微调
模型优化与部署
量化与 TensorRT-LLM 部署
数据集
收集维基百科训练数据
收集 GPT 网络文本数据
可重复性
检查点转换
模型类转换
检查点格式转换
使用 Megatron 的项目

# Megatron Overview
This repository comprises two essential components: **Megatron-LM** and **Megatron-Core**. Megatron-LM serves as a research-oriented framework leveraging Megatron-Core for large language model (LLM) training. Megatron-Core, on the other hand, is a library of GPU optimized training techniques that comes with formal product support including versioned APIs and regular releases. You can use Megatron-Core alongside Megatron-LM or [Nvidia NeMo Framework](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/nemo_megatron/mcore_customization.html) for an end-to-end and cloud-native solution. Alternatively, you can integrate Megatron-Core's building blocks into your preferred training framework.
Megatron 概述
该存储库包含两个重要组件：Megatron-LM 和 Megatron-Core。Megatron-LM 是一个以研究为导向的框架，利用 Megatron-Core 进行大规模语言模型（LLM）训练。而 Megatron-Core 则是一个开源的 PyTorch 库，包含了 GPU 优化的训练技术和前沿的系统级优化。它将这些技术抽象为可组合和模块化的 API，允许开发者和模型研究人员在 NVIDIA 加速计算基础设施上灵活地训练自定义的 Transformer 模型。该库兼容所有 NVIDIA Tensor Core GPU，包括对 NVIDIA Hopper 架构 的 FP8 加速支持。
Megatron-Core 可以与 NVIDIA NeMo 结合使用，提供企业级 AI 平台解决方案。或者，您也可以通过原生 PyTorch 训练循环来探索 Megatron-Core 的功能。访问 Megatron-Core 文档 以了解更多信息。


## Megatron-LM
First introduced in 2019, Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198)) sparked a wave of innovation in the AI community, enabling researchers and developers to utilize the underpinnings of this library to further LLM advancements. Today, many of the most popular LLM developer frameworks have been inspired by and built directly leveraging the open-source Megatron-LM library, spurring a wave of foundation models and AI startups. Some of the most popular LLM frameworks built on top of Megatron-LM include [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [HuggingFace Accelerate](https://github.com/huggingface/accelerate), and [NVIDIA NeMo Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/). A list of projects that have directly used Megatron can be found [here](#projects-using-megatron).
Megatron-LM
最早于 2019 年推出，Megatron (1, 2, 和 3) 在 AI 社区掀起了一波创新浪潮，使研究人员和开发人员能够利用该库的基础来进一步推动 LLM 的发展。如今，许多最流行的 LLM 开发者框架都受到并直接基于开源的 Megatron-LM 库构建，推动了基础模型和 AI 初创公司的发展浪潮。一些基于 Megatron-LM 构建的最受欢迎的 LLM 框架包括 Colossal-AI、HuggingFace Accelerate 和 NVIDIA NeMo Framework。可以直接使用 Megatron 的项目列表可以在 这里 找到。


## Megatron-Core
Megatron-Core is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure. This library is compatible with all NVIDIA Tensor Core GPUs, including FP8 acceleration support for [NVIDIA Hopper architectures](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/). 
Megatron-Core 是一个基于 PyTorch 的开源库，包含 GPU 优化技术和尖端的系统级优化。它将这些技术抽象为可组合和模块化的 API，允许开发者和模型研究人员在 NVIDIA 加速计算基础设施上灵活地进行大规模自定义 Transformer 训练。该库兼容所有 NVIDIA Tensor Core GPU，包括对 NVIDIA Hopper 架构 的 FP8 加速支持。

Megatron-Core offers core building blocks such as attention mechanisms, transformer blocks and layers, normalization layers, and embedding techniques. Additional functionality like activation recomputation, distributed checkpointing is also natively built-in to the library. The building blocks and functionality are all GPU optimized, and can be built with advanced parallelization strategies for optimal training speed and stability on NVIDIA Accelerated Computing Infrastructure. Another key component of the Megatron-Core library includes advanced model parallelism techniques (tensor, sequence, pipeline, context, and MoE expert parallelism). 
Megatron-Core 提供核心构建块，如注意力机制、Transformer 块和层、归一化层以及嵌入技术。像激活重计算、分布式检查点等功能也原生内置在库中。这些构建块和功能都经过 GPU 优化，并且可以通过先进的并行化策略构建，以在 NVIDIA 加速计算基础设施上实现最佳的训练速度和稳定性。Megatron-Core 库的另一个关键组件包括高级模型并行技术（张量、序列、流水线、上下文和 MoE 专家并行）。

Megatron-Core can be used with [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/), an enterprise-grade AI platform. Alternatively, you can explore Megatron-Core with the native PyTorch training loop [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples). Visit [Megatron-Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) to learn more.
Megatron-Core 可以与 NVIDIA NeMo 结合使用，后者是一个企业级 AI 平台。或者，您也可以通过原生 PyTorch 训练循环探索 Megatron-Core 的功能，详情请参见 此处。访问 Megatron-Core 文档 以了解更多内容。


# Training Speed and Scalability 训练速度与可扩展性
Our codebase is capable of efficiently training large language models (i.e., models with hundreds of billions of parameters) with both model and data parallelism. To demonstrate how our software scales with multiple GPUs and model sizes, we consider GPT models ranging from 2 billion parameters to 462 billion parameters. All models use a vocabulary size of 131,072 and a sequence length of 4096. We vary hidden size, number of attention heads, and number of layers to arrive at a specific model size. As the model size increases, we also modestly increase batch size. Our experiments use up to 6144 [H100](https://www.nvidia.com/en-us/data-center/h100/) GPUs. We perform fine-grained overlapping of data-parallel (`--overlap-grad-reduce --overlap-param-gather`), tensor-parallel (`--tp-comm-overlap`) and pipeline-parallel communication (enabled by default) with computation to improve scalability. The reported throughputs are measured for end-to-end training and include all operations including data loading, optimizer steps, communication, and even logging. Note that we did not train these models to convergence.
我们的代码库能够高效地训练超大规模语言模型（例如，拥有数百亿参数的模型），同时支持模型并行和数据并行。为了展示我们软件在多个 GPU 和不同模型规模下的扩展能力，我们测试了从 20 亿参数到 4620 亿参数的 GPT 模型。所有模型的词汇表大小为 131,072，序列长度为 4096。我们通过调整隐藏层大小、注意力头数以及层数来达到特定的模型规模。随着模型规模的增加，我们也适度增加了批量大小。实验使用了多达 6144 个 H100 GPU。我们通过精细地重叠数据并行（--overlap-grad-reduce --overlap-param-gather）、张量并行（--tp-comm-overlap）和流水线并行通信（默认启用）与计算操作，提升了扩展性能。报告的吞吐量是端到端训练的测量结果，包括所有操作，如数据加载、优化步骤、通信甚至日志记录。请注意，我们并未将这些模型训练至收敛。

![Model table](images/model_table.png)

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.
我们的弱扩展结果显示了超线性扩展（MFU 从最小模型的 41% 提高到最大模型的 47%-48%），这是因为更大的 GEMM 具有更高的算术强度，因此执行效率更高。

![Weak scaling](images/weak_scaling.png)

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.
我们还对标准 GPT-3 模型（我们的版本由于更大的词汇表略高于 1750 亿参数）进行了强扩展测试，从 96 个 H100 GPU 扩展到 4608 个 GPU，整个过程中保持相同的批量大小 1152 序列。在更大规模下，通信开销变得更加显著，导致 MFU 从 47% 下降到 42%。

![Strong scaling](images/strong_scaling.png)


# Setup 设置
We strongly recommend using the latest release of [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) with DGX nodes. If you can't use this for some reason, use the latest pytorch, cuda, nccl, and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start) releases.  Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.

You can launch an instance of the PyTorch container and mount Megatron, your dataset, and checkpoints with the following Docker commands:

我们强烈建议使用最新版本的 NGC PyTorch 容器 与 DGX 节点。如果您无法使用此容器，请使用最新的 pytorch、cuda、nccl 和 NVIDIA APEX 版本。数据预处理需要安装 NLTK，但训练、评估或下游任务并不需要。
您可以使用以下 Docker 命令启动 PyTorch 容器，并挂载 Megatron、数据集和检查点：

```
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/megatron:/workspace/megatron -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints nvcr.io/nvidia/pytorch:xx.xx-py3
```

## Downloading Checkpoints 下载检查点
We have provided pretrained [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m) and [GPT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m) checkpoints to evaluate or for finetuning downstream tasks. To access these checkpoints, first [sign up](https://ngc.nvidia.com/signup) for and [setup](https://ngc.nvidia.com/setup/installers/cli) the NVIDIA GPU Cloud (NGC) Registry CLI. Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

Alternatively, you can directly download the checkpoints using:

我们提供了预训练的 BERT-345M 和 GPT-345M 检查点，可用于评估或微调下游任务。要访问这些检查点，请先 注册 并 设置 NVIDIA GPU Cloud (NGC) 注册表 CLI。有关下载模型的更多文档，请参阅 NGC 文档。
或者，您也可以直接下载检查点：

<pre>
BERT-345M-uncased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
BERT-345M-cased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
GPT-345M: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
</pre>

The models require vocabulary files to run. The BERT  WordPiece vocab file can be extracted from Google's pretrained BERT models: 
模型运行需要词汇表文件。BERT WordPiece 词汇表文件可以从 Google 的预训练 BERT 模型中提取：小写，大写。GPT 词汇表文件 和 合并表 可以直接下载。

[uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt), [cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt). The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

# Usage 使用方法

After installation, there are several possible workflows. The most comprehensive is:
1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation
安装后，有几种可能的工作流程。最全面的是：
数据预处理
预训练
微调（对于零样本任务可选）
下游任务评估或文本生成

However, steps 1 and 2 can be replaced by using one of the pretrained models mentioned above.

We've provided several scripts for pretraining both BERT and GPT in the [`examples`](./examples) directory, as well as scripts for both zero-shot and fine-tuned downstream tasks including MNLI, RACE, WikiText103, and LAMBADA evaluation. There is also a script for GPT interactive text generation.

然而，步骤 1 和 2 可以被上述提到的预训练模型替代。
我们在 examples 目录中提供了几个用于 BERT 和 GPT 预训练的脚本，以及用于零样本和微调下游任务（包括 MNLI、RACE、WikiText103 和 LAMBADA 评估）的脚本。还有一个用于 GPT 交互式文本生成的脚本。

# Training 训练
## Data Preprocessing 数据预处理
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
训练数据需要进行预处理。首先，将训练数据放置在松散的 json 格式中，每行一个 json 包含一个文本样本。例如：

<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "第一部分"}
{"src": "互联网", "text": "跳过了懒惰的狗", "type": "Eng", "id": "42", "title": "第二部分"}

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.
可以通过在 preprocess_data.py 中使用 --json-key 标志更改 json 中的 text 字段名称。其他元数据是可选的，不会在训练中使用。

The loose json is then processed into a binary format for training. To convert the json into mmap format use `preprocess_data.py`. An example script to prepare data for BERT training is:
然后将松散的 json 处理成二进制格式进行训练。使用 preprocess_data.py 将 json 转换为 mmap 格式。准备 BERT 训练数据的一个示例脚本如下：

<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab-file bert-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.
输出将是两个文件，本例中名为 my-bert_text_sentence.bin 和 my-bert_text_sentence.idx。在后续的 BERT 训练中，--data-path 指定的是完整路径和新文件名，但不包括文件扩展名。

For T5 use the same preprocessing as BERT, perhaps renaming it to:
对于 T5，使用与 BERT 相同的预处理，或许可以将其重命名为：

<pre>
       --output-prefix my-t5 \
</pre>

Some minor modifications are required for GPT data preprocessing, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
GPT 数据预处理有一些小的修改，即添加合并表、文档结束标记、删除句子分割，并更改分词器类型：

<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT training, use the longer name without the extension as `--data-path`.
这里的输出文件名为 my-gpt2_text_document.bin 和 my-gpt2_text_document.idx。与之前一样，在 GPT 训练中，使用不带扩展名的较长名称作为 --data-path。

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).
更多命令行参数描述请参见源文件 preprocess_data.py。

## BERT Pretraining BERT 预训练 BERT 预训练


The [`examples/bert/train_bert_340m_distributed.sh`](examples/bert/train_bert_340m_distributed.sh) script runs single GPU 345M parameter BERT pretraining. Debugging is the primary use for single GPU training, as the code base and command line arguments are optimized for highly distributed training. Most of the arguments are fairly self-explanatory. By default, the learning rate decays linearly over the training iterations starting at `--lr` to a minimum set by `--min-lr` over `--lr-decay-iters` iterations. The fraction of training iterations used for warmup is set by `--lr-warmup-fraction`. While this is single GPU training, the batch size specified by `--micro-batch-size` is a single forward-backward path batch-size and the code will perform gradient accumulation steps until it reaches `global-batch-size` which is the batch size per iteration. The data is partitioned into a 949:50:1 ratio for training/validation/test sets (default is 969:30:1). This partitioning happens on the fly, but is consistent across runs with the same random seed (1234 by default, or specified manually with `--seed`). We use `train-iters` as the training iterations requested. Alternatively, one can provide `--train-samples` which is total number of samples to train on. If this option is present, then instead of providing `--lr-decay-iters`, one will need to provide `--lr-decay-samples`.

examples/bert/train_bert_340m_distributed.sh 脚本运行单 GPU 345M 参数的 BERT 预训练。
调试是单 GPU 训练的主要用途，因为代码库和命令行参数是针对高度分布式训练优化的。大多数参数都相当直观。
默认情况下，学习率在训练迭代期间从 --lr 开始线性衰减到由 --min-lr 设置的最小值，经过 --lr-decay-iters 次迭代。
用于 warmup 的训练迭代比例由 --lr-warmup-fraction 设置。
虽然是单 GPU 训练，但指定的批量大小 --micro-batch-size 是单次前向-反向传播路径的批量大小，代码将执行梯度累积步骤直到达到 global-batch-size，这是每次迭代的批量大小。数据按 949:50:1 的比例划分为训练/验证/测试集（默认为 969:30:1）。
这种划分是在飞行中完成的，但如果随机种子相同（默认为 1234 或手动指定为 --seed），则在不同运行中是一致的。
我们使用 train-iters 作为请求的训练迭代次数。
或者，可以提供 --train-samples，这是总的训练样本数量。
如果提供了这个选项，则不需要提供 --lr-decay-iters，而是需要提供 --lr-decay-samples。


The logging, checkpoint-saving, and evaluation interval options are specified. Note that the `--data-path` now includes the additional `_text_sentence` suffix added in preprocessing, but does not include the file extensions.
日志记录、检查点保存和评估间隔选项已指定。请注意，--data-path 现在包括预处理中添加的额外 _text_sentence 后缀，但不包括文件扩展名。

Further command line arguments are described in the source file [`arguments.py`](./megatron/training/arguments.py).
更多命令行参数描述请参见源文件 arguments.py。

To run `train_bert_340m_distributed.sh`, make any desired modifications including setting the environment variables for `CHECKPOINT_PATH`, `VOCAB_FILE`, and `DATA_PATH`. Make sure to set these variables to their paths in the container. Then launch the container with Megatron and necessary paths mounted (as explained in [Setup](#setup)) and run the example script.
要运行 train_bert_340m_distributed.sh，请进行任何所需的修改，包括设置环境变量 CHECKPOINT_PATH、VOCAB_FILE 和 DATA_PATH。确保将这些变量设置为容器中的路径。然后按照 Setup 中说明的方式启动带有 Megatron 和必要路径挂载的容器，并运行示例脚本。




## GPT Pretraining GPT 预训练

The `examples/gpt3/train_gpt3_175b_distributed.sh` script runs single GPU 345M parameter GPT pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.
examples/gpt3/train_gpt3_175b_distributed.sh 脚本运行单 GPU 345M 参数的 GPT 预训练。如前所述，单 GPU 训练主要用于调试目的，因为代码是针对分布式训练优化的。

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.
它基本上遵循与前面的 BERT 脚本相同的格式，但有一些显著的区别：使用的分词方案是 BPE（需要合并表和 json 词汇表文件），而不是 WordPiece；模型架构允许更长的序列（注意最大位置嵌入必须大于或等于最大序列长度）；并且 --lr-decay-style 已设置为余弦衰减。请注意，--data-path 现在包括预处理中添加的额外 _text_document 后缀，但不包括文件扩展名。

Further command line arguments are described in the source file [`arguments.py`](./megatron/training/arguments.py).
更多命令行参数描述请参见源文件 arguments.py。

`train_gpt3_175b_distributed.sh` can be launched the same way as described for BERT. Set the env vars and make any other modifications, launch the container with appropriate mounts, and run the script.
More details in [`examples/gpt3/README.md`](./examples/gpt3/README.md)
train_gpt3_175b_distributed.sh 可以按照上述 BERT 描述的方式启动。设置环境变量并进行任何其他修改，启动带有适当挂载的容器，并运行脚本。
 更多详细信息请参见 examples/gpt3/README.md

## T5 Pretraining T5 预训练

Very similar to BERT and GPT, the `examples/t5/train_t5_220m_distributed.sh` script runs single GPU "base" (~220M parameter) T5 pretraining. The primary difference from BERT and GPT is the addition of the following arguments to accommodate the T5 architecture:
与 BERT 和 GPT 非常相似，examples/t5/train_t5_220m_distributed.sh 脚本运行单 GPU "base"（约 220M 参数）的 T5 预训练。与 BERT 和 GPT 预训练相比的主要区别在于添加了以下参数以适应 T5 架构：

* `--kv-channels` sets the inner dimension of the "key" and "value" matrices of all attention mechanisms in the model. For BERT and GPT this defaults to the hidden size divided by the number of attention heads, but can be configured for T5.

* `--ffn-hidden-size` sets the hidden size in the feed-forward networks within a transformer layer. For BERT and GPT this defaults to 4 times the transformer hidden size, but can be configured for T5.

* `--encoder-seq-length` and `--decoder-seq-length` set the sequence length for the encoder and decoder separately.
--kv-channels 设置模型中所有注意力机制的“键”和“值”矩阵的内部维度。对于 BERT 和 GPT，默认值为隐藏层大小除以注意力头数，但对于 T5 可以配置。
--ffn-hidden-size 设置变压器层内前馈网络的隐藏层大小。对于 BERT 和 GPT，默认值为变压器隐藏层大小的 4 倍，但对于 T5 可以配置。
--encoder-seq-length 和 --decoder-seq-length 分别设置编码器和解码器的序列长度。

All of the other arguments remain as they were for BERT and GPT pretraining. Run this example with the same steps described above for the other scripts.
所有其他参数与 BERT 和 GPT 预训练时保持不变。按照上述其他脚本描述的步骤运行此示例。

More details in [`examples/t5/README.md`](./examples/t5/README.md)
更多详细信息请参见 examples/t5/README.md

## Distributed Pretraining 分布式预训练

The `pretrain_{bert,gpt,t5}_distributed.sh` scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables. See the official PyTorch [documentation](https://pytorch.org/docs/stable/elastic/run.html#launcher-api) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the `torchrun` elastic launcher (equivalent to `python -m torch.distributed.run`) are the only additional requirements to adopt distributed training. See any of `pretrain_{bert,gpt,t5}_distributed.sh` for more details.
pretrain_{bert,gpt,t5}_distributed.sh 脚本使用 PyTorch 分布式启动器进行分布式训练。因此，通过正确设置环境变量可以实现多节点训练。请参阅官方 PyTorch 文档 了解更多关于这些 环境变量 的描述。默认情况下，多节点训练使用 nccl 分布式后端。一组额外的参数和使用 PyTorch 分布式模块的 torchrun 弹性启动器（等同于 python -m torch.distributed.run）是采用分布式训练的唯一额外要求。更多细节请参见任何 pretrain_{bert,gpt,t5}_distributed.sh。

We use two types of parallelism: data and model parallelism. Our data parallelism implementation is in `megatron/core/distributed`, and supports overlapping of the gradient reduction with the backward pass when the `--overlap-grad-reduce` command-line option is used.
我们使用两种类型的并行性：数据并行和模型并行。我们的数据并行实现位于 megatron/core/distributed 中，当使用 --overlap-grad-reduce 命令行选项时，支持将梯度缩减与反向传播重叠。

Second, we developed a simple and efficient two-dimensional model-parallel approach. To use the first dimension, tensor model parallelism (splitting execution of a single transformer module over multiple GPUs, see Section 3 of [our paper](https://arxiv.org/pdf/1909.08053.pdf)), add the `--tensor-model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use the second dimension, sequence parallelism, specify `--sequence-parallel`, which also requires tensor model parallelism to be enabled because it splits across the same GPUs (more details in Section 4.2.2 of [our paper](https://arxiv.org/pdf/2205.05198.pdf)).
其次，我们开发了一种简单而高效的二维模型并行方法。要使用第一个维度，张量模型并行（将单个变压器模块的执行分布在多个 GPU 上，详见我们论文的第 3 节），添加 --tensor-model-parallel-size 标志来指定拆分模型的 GPU 数量，以及传递给分布式启动器的参数，如上所述。要使用第二个维度，序列并行，指定 --sequence-parallel，这也需要启用张量模型并行，因为它在同一组 GPU 上进行拆分（更多细节请参见我们论文的第 4.2.2 节）。

To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches, see Section 2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)), use the `--pipeline-model-parallel-size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).
要使用管道模型并行（将变压器模块分成阶段，每个阶段具有相同数量的变压器模块，然后通过将批次分成较小的微型批次来管道化执行，详见我们论文的第 2.2 节），使用 --pipeline-model-parallel-size 标志来指定将模型拆分成的阶段数（例如，将具有 24 层变压器的模型跨 4 个阶段拆分意味着每个阶段获得 6 层变压器）。


We have examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`.
我们提供了如何使用这两种不同形式的模型并行的例子，可以在以 distributed_with_mp.sh 结尾的示例脚本中找到。

Other than these minor changes, the distributed training is identical to the training on a single GPU.
除了这些小改动外，分布式训练与单 GPU 训练完全相同。

The interleaved pipelining schedule (more details in Section 2.2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)) can be enabled using the `--num-layers-per-virtual-pipeline-stage` argument, which controls the number of transformer layers in a virtual stage (by default with the non-interleaved schedule, each GPU will execute a single virtual stage with `NUM_LAYERS / PIPELINE_MP_SIZE` transformer layers). The total number of layers in the transformer model should be divisible by this argument value. Additionally, the number of microbatches in the pipeline (computed as `GLOBAL_BATCH_SIZE / (DATA_PARALLEL_SIZE * MICRO_BATCH_SIZE)`) should be divisible by the `PIPELINE_MP_SIZE` when using this schedule (this condition is checked in an assertion in the code). The interleaved schedule is not supported for pipelines with 2 stages (`PIPELINE_MP_SIZE=2`).
交错管道调度（更多细节请参见我们论文的第 2.2.2 节）可以通过 --num-layers-per-virtual-pipeline-stage 参数启用，该参数控制虚拟阶段中的变压器层数量（在非交错调度的情况下，默认每个 GPU 执行一个虚拟阶段，包含 NUM_LAYERS / PIPELINE_MP_SIZE 个变压器层）。变压器模型中的总层数应能被该参数值整除。此外，管道中的微型批次数量（计算为 GLOBAL_BATCH_SIZE / (DATA_PARALLEL_SIZE * MICRO_BATCH_SIZE)）应能被 PIPELINE_MP_SIZE 整除，当使用此调度时（代码中有一个断言检查此条件）。交错调度不支持 2 阶段的管道（PIPELINE_MP_SIZE=2）。

## Activation Checkpointing and Recomputation 激活检查点和重计算

To reduce GPU memory usage when training a large model, we support various forms of activation checkpointing and recomputation. Instead of all activations being stored in memory to be used during backprop, as was traditionally the case in deep learning models, only activations at certain "checkpoints" in the model are retained (or stored) in memory, and the other activations are recomputed on-the-fly when needed for backprop. Note that this kind of checkpointing, *activation* checkpointing, is very different from the checkpointing of model parameters and optimizer state, which is mentioned elsewhere.
为了在训练大型模型时减少 GPU 内存使用，我们支持各种形式的激活检查点和重计算。
传统深度学习模型中，所有激活值都会存储在内存中以供反向传播使用，而在激活检查点中，只有某些“检查点”处的激活值保留在内存中，其他激活值则在需要时重新计算。
请注意，这种检查点，即激活检查点，与模型参数和优化器状态的检查点完全不同，后者在其他地方有所提及。


We support two levels of recompute granularity: `selective` and `full`. Selective recomputation is the default and is recommended in almost all cases. This mode retains in memory the activations that take less memory storage space and are more expensive to recompute and recomputes the activations that take more memory storage space but are relatively inexpensive to recompute. See [our paper](https://arxiv.org/pdf/2205.05198) for details. You should find that this mode maximizes performance while minimizing the memory required to store activations. To enable selective activation recompute simply use `--recompute-activations`.
我们支持两种重计算粒度级别：selective 和 full。
选择性重计算是默认值，几乎在所有情况下都推荐使用。
在这种模式下，内存中保留的是占用较少存储空间且重新计算成本较高的激活值，而重新计算的是占用较多存储空间但相对便宜的激活值。详见我们的论文。
您会发现这种模式在最大化性能的同时最大限度地减少了存储激活值所需的内存。
要启用选择性激活重计算，只需使用 --recompute-activations。

For cases where memory is very limited, `full` recompute saves just the inputs to a transformer layer, or a group, or block, of transformer layers, and recomputes everything else. To enable full activation recompute use `--recompute-granularity full`. When using `full` activation recompute, there are two methods: `uniform` and `block`, chosen using the `--recompute-method` argument.
对于内存非常有限的情况，full 重计算仅存储变压器层或变压器层组或块的输入激活值，并重新计算所有其他激活值。
要启用完全激活重计算，请使用 --recompute-granularity full。当使用 full 激活重计算时，有两种方法：uniform 和 block，通过 --recompute-method 参数选择。

* The `uniform` method uniformly divides the transformer layers into groups of layers (each group of size `--recompute-num-layers`) and stores the input activations of each group in memory. The baseline group size is 1 and, in this case, the input activation of each transformer layer is stored. When the GPU memory is insufficient, increasing the number of layers per group reduces the memory usage, enabling a bigger model to be trained. For example, when `--recompute-num-layers` is set to 4, only the input activation of each group of 4 transformer layers is stored.
uniform 方法将变压器层均匀地分成若干组（每组大小为 --recompute-num-layers），并将每组的输入激活值存储在内存中。基线组大小为 1，在这种情况下，每个变压器层的输入激活值都被存储。当 GPU 内存不足时，增加每组层数可以减少内存使用，从而能够训练更大的模型。例如，当 --recompute-num-layers 设置为 4 时，只存储每组 4 层变压器的输入激活值。

* The `block` method recomputes the input activations of a specific number (given by `--recompute-num-layers`) of individual transformer layers per pipeline stage and stores the input activations of the remaining layers in the pipeline stage. Reducing `--recompute-num-layers` results in storing the input activations to more transformer layers, which reduces the activation recomputation required in the backprop, thus improving training performance while increasing memory usage. For example, when we specify 5 layers to recompute of 8 layers per pipeline stage, the input activations of only the first 5 transformer layers are recomputed in the backprop step while the input activations for the final 3 layers are stored. `--recompute-num-layers` can be incrementally increased until the amount of memory storage space required is just small enough to fit in the available memory, thereby both maximally utilizing memory and maximizing performance.
block 方法重新计算每个管道阶段中特定数量（由 --recompute-num-layers 给出）的单个变压器层的输入激活值，并存储该管道阶段中剩余层的输入激活值。减少 --recompute-num-layers 会导致存储更多变压器层的输入激活值，从而减少反向传播所需的激活重计算，提高训练性能，但会增加内存使用。例如，当我们指定每个管道阶段的 8 层中有 5 层进行重计算时，在反向传播步骤中只重新计算前 5 层变压器的输入激活值，而最后 3 层的输入激活值被存储。可以逐步增加 --recompute-num-layers，直到所需的存储空间刚好足够放入可用内存中，从而既充分利用内存又最大化性能。


## Distributed Optimizer 分布式优化器

Usage: `--use-distributed-optimizer`. Compatible with all model and data types.
用法：--use-distributed-optimizer。兼容所有模型和数据类型。

The distributed optimizer is a memory savings technique, whereby the optimizer state is evenly distributed across data parallel ranks (versus the traditional method of replicating the optimizer state across data parallel ranks). As described in [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), our implementation distributes all optimizer state that does not overlap with the model state. For example, when using fp16 model params, the distributed optimizer maintains its own separate copy of fp32 main params & grads, which are distributed across DP ranks. When using bf16 model params, however, the distributed optimizer's fp32 main grads are the same as the model's fp32 grads, and so the grads in this case are not distributed (although the fp32 main params are still distributed, as they are separate from the bf16 model params).
分布式优化器是一种节省内存的技术，其中优化器状态在数据并行等级之间均匀分布（与传统的在数据并行等级之间复制优化器状态的方法相反）。
正如 ZeRO: Memory Optimizations Toward Training Trillion Parameter Models 中所描述的，我们的实现分布了所有不与模型状态重叠的优化器状态。
例如，当使用 fp16 模型参数时，分布式优化器维护其自己的单独的 fp32 主参数和梯度副本，这些副本在 DP 等级之间分布。
当使用 bf16 模型参数时，分布式优化器的 fp32 主梯度与模型的 fp32 梯度相同，因此在这种情况下梯度不分布（尽管 fp32 主参数仍然分布，因为它们与 bf16 模型参数不同）。

Theoretical memory savings vary depending on the combination of the model's param dtype and grad dtype. In our implementation, the theoretical number of bytes per parameter is (where 'd' is the data parallel size):
理论上的内存节省因模型参数的数据类型和梯度数据类型的组合而异。在我们的实现中，每参数的理论字节数是（其中 'd' 是数据并行大小）：

| | Non-distributed optim | Distributed optim |
|-|-|-|
| fp16 param, fp16 grads | 20 | 4 + 16/d |
| bf16 param, fp32 grads | 18 | 6 + 12/d |
| fp32 param, fp32 grads | 16 | 8 + 8/d |
非分布式优化器	分布式优化器
fp16 参数，fp16 梯度	20	4 + 16/d
bf16 参数，fp32 梯度	18	6 + 12/d
fp32 参数，fp32 梯度	16	8 + 8/d

As with regular data parallelism, overlapping of the gradient reduction (in this case, a reduce-scatter) with the backward pass can be facilitated using the `--overlap-grad-reduce` flag. Additionally, overlapping of the parameter all-gather can be overlapped with the forward pass using `--overlap-param-gather`.
与常规数据并行一样，梯度缩减（在这种情况下是 reduce-scatter）与反向传播的重叠可以通过 --overlap-grad-reduce 标志来促进。此外，参数 all-gather 与前向传播的重叠可以通过 --overlap-param-gather 来实现。


## FlashAttention

Usage: `--use-flash-attn`. Support attention head dimensions at most 128.

[FlashAttention](https://github.com/HazyResearch/flash-attention) is a fast and
memory-efficient algorithm to compute exact attention. It speeds up model
training and reduces memory requirement.

To install FlashAttention:
```sh
pip install flash-attn
```

FlashAttention
用法：--use-flash-attn。支持最多 128 的注意力头维度。
FlashAttention 是一种快速且内存高效的算法，用于计算精确的注意力。它可以加速模型训练并减少内存需求。
安装 FlashAttention：
pip install flash-attn

## GPT-3 Example GPT-3 示例

In `examples/gpt3/train_gpt3_175b_distributed.sh` we have provided an example of how to configure Megatron to train [GPT-3](https://arxiv.org/abs/2005.14165) with 175 billion parameters on 1024 GPUs. The script is designed for [slurm](https://slurm.schedmd.com/documentation.html) with [pyxis](https://github.com/NVIDIA/pyxis) plugin but can be easily adopted to any other scheduler. It uses 8-way tensor parallelism and 16-way pipeline parallelism. With options `global-batch-size 1536` and `rampup-batch-size 16 16 5859375`, the training will start with global batch size 16 and linearly increase the global batch size to 1536 over 5,859,375 samples with incremental steps 16. The training dataset can be either a single set or a multiple datasets combined with a set of weights.
在 examples/gpt3/train_gpt3_175b_distributed.sh 中，我们提供了一个如何配置 Megatron 来训练 GPT-3 的示例，该模型有 1750 亿个参数，在 1024 个 GPU 上训练。该脚本专为 slurm 和 pyxis 插件设计，但可以轻松适配到任何其他调度器。它使用 8 路张量并行和 16 路流水线并行。通过选项 global-batch-size 1536 和 rampup-batch-size 16 16 5859375，训练将从全局批量大小 16 开始，并在线性增加到 1536 的过程中覆盖 5,859,375 个样本，增量步长为 16。训练数据集可以是单一数据集，也可以是由权重组合的多个数据集。

With full global batch size of 1536 on 1024 A100 GPUs, each iteration takes around 32 seconds resulting in 138 teraFLOPs per GPU which is 44% of the theoretical peak FLOPs.
在 1024 个 A100 GPU 上使用全全局批量大小 1536 时，每次迭代大约需要 32 秒，每块 GPU 的计算能力为 138 teraFLOPs，占理论峰值 FLOPs 的 44%。

## Retro and InstructRetro


Retro [(Borgeaud et al., 2022)](https://arxiv.org/abs/2112.04426) is an autoregressive decoder-only language model (LM) pretrained with retrieval-augmentation.
Retro features practical scalability to support large-scale pretraining from scratch by retrieving from trillions of tokens.
Pretraining with retrieval provides a more efficient storage mechanism of factual knowledge, when compared to storing factual knowledge implicitly within the network's parameters, thus largely reducing model parameters while achieving lower perplexity than standard GPT.
Retro also provides the flexibility to update the
knowledge stored in LMs [(Wang et al., 2023a)](https://arxiv.org/abs/2304.06762)
by updating the retrieval database without training LMs again.
Retro (Borgeaud et al., 2022) 是一种预训练的自回归解码器-only 语言模型（LM），通过检索增强进行预训练。
 Retro 特点是实用的可扩展性，支持从头开始通过检索万亿个令牌进行大规模预训练。
 通过检索进行预训练提供了一种更有效的事实知识存储机制，相比于在神经网络参数中隐式存储事实知识，这种方法大大减少了模型参数，同时实现了比标准 GPT 更低的困惑度。
 Retro 还提供了更新存储在语言模型中的知识的灵活性 (Wang et al., 2023a)
 通过更新检索数据库而无需再次训练语言模型。

InstructRetro [(Wang et al., 2023b)](https://arxiv.org/abs/2310.07713) further scales up the size of Retro to 48B, featuring the largest LLM pretrained with retrieval (as of December 2023).
The obtained foundation model, Retro 48B, largely outperforms the GPT counterpart in terms of perplexity.
With instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on downstream tasks in the zero-shot setting. Specifically, the average improvement of InstructRetro is 7% over its GPT counterpart across 8 short-form QA tasks, and 10% over GPT across 4 challenging long-form QA tasks. We also find that one can ablate the encoder from InstructRetro architecture and directly use the InstructRetro decoder backbone as GPT, while achieving comparable results.
InstructRetro (Wang et al., 2023b) 进一步将 Retro 的规模扩大到 48B，成为迄今为止最大的预训练检索语言模型（截至 2023 年 12 月）。
 由此获得的基础模型 Retro 48B 在困惑度方面大幅优于其 GPT 对手。
 通过对 Retro 进行指令调优，InstructRetro 在下游任务的零样本设置中展示了相对于指令调优的 GPT 显著改进。具体来说，InstructRetro 在 8 个短格式问答任务上的平均提升为 7%，在 4 个具有挑战性的长格式问答任务上的提升为 10%。我们还发现可以去除 InstructRetro 架构中的编码器，并直接使用 InstructRetro 解码器骨干作为 GPT，同时实现可比的结果。

In this repo, we provide an end-to-end reproduction guide to implement Retro and InstructRetro, covering
- **Retrieval database construction**, which supports billions or even trillions of tokens as a large-scale retrieval database.
- **Pretraining with retrieval**, which supports pretraining from scratch and pretraining from a pretrained GPT model (Retro-fitting).
- **Instruction tuning**, where we provide an open-source instruction tuning dataset and the training recipe for instruction tuning on Retro.
- **Downstream task evaluation**, where we provide the text generation and evaluation scripts for zero-shot question answering tasks.
在这个仓库中，我们提供了一个端到端的复现指南，用于实现 Retro 和 InstructRetro，涵盖：
检索数据库构建，支持数十亿甚至数万亿个令牌作为大规模检索数据库。
预训练检索，支持从头开始预训练和从预训练的 GPT 模型进行预训练（Retro-fitting）。
指令调优，我们提供了一个开源的指令调优数据集和对 Retro 进行指令调优的训练配方。
下游任务评估，我们提供了文本生成和零样本问答任务的评估脚本。

See [tools/retro/README.md](tools/retro/README.md) for a detailed overview.
请参阅 tools/retro/README.md 获取详细概述。

## Mamba-based Language Models

See [examples/mamba](./examples/mamba) for details.

基于 Mamba 的语言模型
详情请参阅 examples/mamba。

<!--
## REALM Pipeline
We are working on implementing the [REALM](https://arxiv.org/pdf/2002.08909.pdf) system. The following sections (will) reflect the three stages of training it. For now it's just the ICT code.
Loosely, they are pretraining the retriever modules, then jointly training the language model and the retriever, and then finetuning a question answering head on the language model with fixed retriever.

### Inverse Cloze Task (ICT) Pretraining
1. Have a corpus in loose JSON format with the intention of creating a collection of fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block but also multiple blocks per document.
Run `tools/preprocess_data.py` to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. For the original REALM system, we construct two datasets, one with the title of every document, and another with the body.
Refer to the following script
<pre>
python preprocess_data.py \
    --input /path/to/corpus.json \
    --json-keys text title \
    --split-sentences \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /path/to/vocab.txt \
    --output-prefix corpus_indexed \
    --workers 5  # works well for 10 CPU cores. Scale up accordingly.
</pre>

2. Use a custom samples mapping function in place of `megatron/legacy/data/realm_dataset_utils.get_block_samples_mapping` if required. To do this, you will need to implement a new function in C++ inside of `megatron/core/datasets/helpers.cpp`. The samples mapping data structure is used to select the data that will constitute every training sample in advance of the training loop.
 The samples mapping is responsible for holding all of the required metadata needed to construct the sample from one or more indexed datasets. In REALM, the samples mapping contains the start and end sentence indices, as well as the document index (to find the correct title for a body) and a unique ID for every block.
3. Pretrain a BERT language model using `pretrain_bert.py`, with the sequence length equal to the block size in token ids. This model should be trained on the same indexed dataset that is used to supply the blocks for the information retrieval task.
In REALM, this is an uncased bert base model trained with the standard hyperparameters.
4. Use `pretrain_ict.py` to train an `ICTBertModel` which uses two BERT-based encoders to encode queries and blocks to perform retrieval with.
The script below trains the ICT model from REALM. It references a pretrained BERT model (step 3) in the `--bert-load` argument. The batch size used in the paper is 4096, so this would need to be run with data parallel world size 32.
<pre>
python pretrain_ict.py \
    --num-layers 12 \
    --num-attention-heads 12 \
    --hidden-size 768 \
    --batch-size 128 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-head-size 128 \
    --train-iters 100000 \
    --bert-load /path/to/pretrained_bert \
    --load checkpoints \
    --save checkpoints \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --vocab-file /path/to/vocab.txt \
    --lr 0.0001 \
    --num-workers 2 \
    --lr-decay-style linear \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --save-interval 3000 \
    --query-in-block-prob 0.1 \
    --fp16

</pre>

### Building an Index of Block Embeddings
After having trained an ICT model, you can now embed an entire dataset of blocks by creating a `BlockData` structure. After that has been saved, you can load it
and wrap it with a `FaissMIPSIndex` to do fast similarity search which is key in the learned information retrieval pipeline. The initial index can be built with the following script, meant to be run in an interactive session. It can leverage multiple GPUs on multiple nodes to index large datasets much more quickly.

<pre>
python tools/create_doc_index.py \
    --num-layers 12 \
    --hidden-size 768 \
    --ict-head-size 128 \
    --num-attention-heads 12 \
    --batch-size 128 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-load /path/to/pretrained_ict \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --block-data-path embedded_blocks.pkl \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128 \
    --vocab-file /path/to/vocab.txt \
    --num-workers 2 \
    --fp16
</pre>

-->

## Mixture of Experts 混合专家模型 (Mixture of Experts, MoE)
MoE (Mixture of Experts) is a powerful LLM architecture implemented in the Megatron-Core framework, designed to enhance the efficiency and scalability of large language models. It leverages **Expert Parallelism**, allowing multiple experts to be distributed across different workers, where each worker processes distinct batches of training samples. This method significantly increases computational throughput, enabling models to achieve high performance metrics, such as 47% MFU during BF16 training for 8x7B on H100.
MoE（混合专家模型）是一种在 Megatron-Core 框架中实现的强大 LLM 架构，旨在提高大规模语言模型的效率和可扩展性。它利用了 专家并行性，允许多个专家分布在不同的工作节点上，每个工作节点处理不同的训练样本批次。这种方法显著提高了计算吞吐量，使得模型能够在 H100 上进行 BF16 训练时达到 47% 的 MFU（例如 8x7B 模型）。

Key Features of MoE:
- **Parallelism Techniques**: MoE combines various parallelism strategies, including Expert Parallelism, Data Parallelism, Tensor Parallelism, Sequence Paralleism, Pipeline Parallelism, and Context Parallelism. This combination allows for handling larger model variants effectively.
- **Router and Load Balancing**: The system employs advanced routing mechanisms like the Top-K router and utilizes load balancing algorithms to optimize token distribution among experts.
- **Performance Optimizations**: Techniques such as GroupedGEMM and FP8 training enhance the efficiency of MoE models, particularly when multiple experts are involved.
- **Token Dispatch Mechanism**: MoE supports both dropless and token drop strategies to manage token distribution effectively across experts.
MoE 的主要特点：
并行技术：MoE 结合了多种并行策略，包括专家并行、数据并行、张量并行、序列并行、流水线并行和上下文并行。这种组合可以有效处理更大的模型变体。
路由与负载均衡：系统采用先进的路由机制（如 Top-K 路由器）并使用负载均衡算法来优化令牌在专家之间的分布。
性能优化：GroupedGEMM 和 FP8 训练等技术增强了 MoE 模型的效率，尤其是在涉及多个专家的情况下。
令牌分发机制：MoE 支持无丢弃和令牌丢弃策略，以有效地管理专家之间的令牌分布。

For a comprehensive overview of MoE training configurations and optimizations, please refer to the detailed README located at [megatron/core/transformer/moe/README.md](./megatron/core/transformer/moe/README.md).
有关 MoE 训练配置和优化的详细概述，请参阅 megatron/core/transformer/moe/README.md。

# Evaluation and Tasks 评估与任务

We provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.
我们提供了几个命令行参数（详见以下脚本），用于处理各种零样本和微调的下游任务。此外，您还可以从预训练检查点对其他语料库进行微调。为此，只需添加 --finetune 标志，并调整原始训练脚本中的输入文件和训练参数。迭代计数将重置为零，优化器和内部状态也会重新初始化。如果微调因任何原因中断，请确保在继续之前移除 --finetune 标志，否则训练将从头开始。

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on fewer GPUs in downstream tasks. The following script accomplishes this. This example reads in a GPT model with 4-way tensor and 4-way pipeline model parallelism and writes out a model with 2-way tensor and 2-way pipeline model parallelism.
由于评估所需的内存远少于训练，因此将并行训练的模型合并到更少的 GPU 上进行下游任务可能是有利的。以下脚本完成了这一操作。该示例读取一个具有 4 路张量并行和 4 路流水线并行的 GPT 模型，并输出一个具有 2 路张量并行和 2 路流水线并行的模型。

<pre>
python tools/checkpoint/convert.py \
        --model-type GPT \
        --load-dir checkpoints/gpt3_tp4_pp4 \
        --save-dir checkpoints/gpt3_tp2_pp2 \
        --target-tensor-parallel-size 2 \
        --target-pipeline-parallel-size 2

</pre>

Several downstream tasks are described for both GPT and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.
以下描述了 GPT 和 BERT 模型的几个下游任务。它们可以通过与训练脚本相同的更改在分布式和模型并行模式下运行。

## GPT Text Generation GPT 文本生成

We have included a simple REST server to use for text generation in `tools/run_text_generation_server.py`. You run it much like you would start a pretraining job, specifying an appropriate pretrained checkpoint. There are also few optional parameters: `temperature`, `top-k`and `top-p`. See `--help` or the source file for more information. See [examples/inference/run_text_generation_server_345M.sh](examples/inference/run_text_generation_server_345M.sh) for an example of how to run the server.
我们在 tools/run_text_generation_server.py 中包含了一个简单的 REST 服务器，用于文本生成。
您可以像启动预训练任务一样运行它，指定适当的预训练检查点。
还有一些可选参数：temperature、top-k 和 top-p。
更多信息请参见 --help 或源文件。
有关如何运行服务器的示例，请参见 examples/inference/run_text_generation_server_345M.sh。

Once the server is running you can use `tools/text_generation_cli.py` to query it, it takes one argument which is the host the server is running on.
一旦服务器运行，您可以使用 tools/text_generation_cli.py 查询它，该工具接受一个参数，即服务器运行的主机地址。

<pre>
tools/text_generation_cli.py localhost:5000
</pre>

You can also use CURL or any other tools to query the server directly:
您也可以使用 CURL 或其他工具直接查询服务器：

<pre>
curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["Hello world"], "tokens_to_generate":1}'
</pre>

See [megatron/inference/text_generation_server.py](megatron/inference/text_generation_server.py) for more API options.
更多 API 选项请参见 megatron/inference/text_generation_server.py。

### Detoxify GPT via Self-generation 通过自生成净化 GPT
We include an example in `examples/academic_paper_scripts/detxoify_lm/` to detoxify language models by leveraging the generative power of language models.
我们在 examples/academic_paper_scripts/detxoify_lm/ 中包含了一个示例，通过利用语言模型的生成能力来净化语言模型。

See [examples/academic_paper_scripts/detxoify_lm/README.md](examples/academic_paper_scripts/detxoify_lm/README.md) for step-by-step tutorials on how to perform domain-adaptive training and detoxify LM using self-generated corpus.
有关如何执行领域适应训练和使用自生成语料库净化 LM 的分步教程，请参见 examples/academic_paper_scripts/detxoify_lm/README.md。

## GPT Evaluation GPT 评估
We include example scripts for GPT evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.
我们包含了用于 GPT 评估的示例脚本，涵盖 WikiText 困惑度评估和 LAMBADA 完形填空准确率。

### WikiText Perplexity Evaluation WikiText 困惑度评估
For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.
为了与先前的工作进行公平比较，我们在单词级 WikiText-103 测试数据集 上评估困惑度，并根据使用子词分词器时的标记变化适当地计算困惑度。

We use the following command to run WikiText-103 evaluation on a 345M parameter model.
我们使用以下命令对 345M 参数模型运行 WikiText-103 评估：

<pre>
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>


### LAMBADA Cloze Accuracy   LAMBADA 完形填空准确率
To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceding tokens) we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).
为了计算 LAMBADA 完形填空准确率（给定前面的标记预测最后一个标记的准确性），我们使用了 LAMBADA 数据集 的去标记化、处理版本。

We use the following command to run LAMBADA evaluation on a 345M parameter model. Note that the `--strict-lambada` flag should be used to require whole word matching. Ensure that `lambada` is part of the file path.
我们使用以下命令对 345M 参数模型运行 LAMBADA 评估。请注意，应使用 --strict-lambada 标志以要求全词匹配。确保 lambada 是文件路径的一部分。

<pre>
TASK="LAMBADA"

VALID_DATA=&#60;lambada path&#62;.json
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m
COMMON_TASK_ARGS=&#60;same as those in <a href="#wikitext-perplexity-evaluation">WikiText Perplexity Evaluation</a> above&#62;

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --strict-lambada \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>

Further command line arguments are described in the source file [`main.py`](./tasks/main.py)
更多命令行参数描述请参见源文件 main.py。

## BERT Task Evaluation BERT 任务评估
### RACE Evaluation    RACE 评估
The following script finetunes the BERT model for evaluation on the [RACE dataset](http://www.cs.cmu.edu/~glai1/data/race/). The `TRAIN_DATA` and `VALID_DATA` directory contain the RACE dataset as separate `.txt` files. Note that for RACE, the batch size is the number of RACE query's to evaluate. Since each RACE query has four samples, the effective batch size passed through the model will be four times the batch size specified on the command line.
以下脚本对 RACE 数据集 进行 BERT 模型的微调。TRAIN_DATA 和 VALID_DATA 目录包含 RACE 数据集作为单独的 .txt 文件。注意，对于 RACE，批量大小是 RACE 查询的数量。由于每个 RACE 查询有四个样本，传递给模型的有效批量大小将是命令行中指定的批量大小的四倍。

<pre>
TRAIN_DATA="data/RACE/train/middle"
VALID_DATA="data/RACE/dev/middle \
            data/RACE/dev/high"
VOCAB_FILE=bert-vocab.txt
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/bert_345m_race
COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 10 \
                      --weight-decay 1.0e-1"

python tasks/main.py \
       --task RACE \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 3 \
       --micro-batch-size 4 \
       --lr 1.0e-5 \
       --lr-warmup-fraction 0.06
</pre>

### MNLI Evaluation    MNLI 评估
The following script finetunes the BERT model for evaluation with the [MultiNLI sentence pair corpus](https://www.nyu.edu/projects/bowman/multinli/). Because the matching tasks are quite similar, the script can be quickly tweaked to work with the [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset) (QQP) dataset as well.
以下脚本对 MultiNLI 句对语料库 进行 BERT 模型的微调。由于匹配任务非常相似，该脚本可以快速调整以适用于 Quora 问题对 (QQP) 数据集。

<pre>

TRAIN_DATA="data/glue_data/MNLI/train.tsv"
VALID_DATA="data/glue_data/MNLI/dev_matched.tsv \
            data/glue_data/MNLI/dev_mismatched.tsv"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m_mnli
COMMON_TASK_ARGS=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;
COMMON_TASK_ARGS_EXT=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;

python tasks/main.py \
       --task MNLI \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 5 \
       --micro-batch-size 8 \
       --lr 5.0e-5 \
       --lr-warmup-fraction 0.065
</pre>

## Llama-2 Inference and Finetuning  Llama-2 推理和微调

The Llama-2 [family of models](https://ai.meta.com/llama/) are an open-source set of pretrained & finetuned (for chat) models that have achieved strong results across a wide set of benchmarks. At the time of release, Llama-2 models achieved among the best results for open-source models, and were competitive with the closed-source GPT-3.5 model (see https://arxiv.org/pdf/2307.09288.pdf).
Llama-2 模型系列 是一组开源的预训练和微调（用于聊天）模型，在广泛的基准测试中取得了优异的结果。在发布时，Llama-2 模型在开源模型中表现最佳，并且与闭源的 GPT-3.5 模型具有竞争力（参见 https://arxiv.org/pdf/2307.09288.pdf）。

The Llama-2 checkpoints can be loaded into Megatron for inference and finetuning. See documentation [here](docs/llama_mistral.md).
Llama-2 检查点可以加载到 Megatron 中进行推理和微调。更多文档请参见 此处。

# Model Optimization and Deployment
Megatron-Core (MCore) `GPTModel` family supports advanced quantization algorithms and high-performance inference through TensorRT-LLM.
模型优化与部署
Megatron-Core (MCore) 的 GPTModel 系列支持高级量化算法和通过 TensorRT-LLM 实现的高性能推理。

## Quantization and TensorRT-LLM Deployment
See [Megatron Model Optimization and Deployment](examples/inference/quantization/README.md) for `llama2` and `nemotron3` examples.
量化与 TensorRT-LLM 部署
有关 llama2 和 nemotron3 示例的详细信息，请参见 Megatron 模型优化与部署。

# Datasets
We do not host any datasets for GPT or BERT training, however, we detail their collection so that our results may be reproduced.
数据集
我们不托管任何用于 GPT 或 BERT 训练的数据集，但我们详细说明了它们的收集过程，以便我们的结果可以被复现。

## Collecting Wikipedia Training Data
We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."
收集维基百科训练数据
我们建议遵循 Google 研究团队推荐的维基百科数据提取流程：“推荐的预处理步骤是下载最新转储，使用 WikiExtractor.py 提取文本，然后应用必要的清理将其转换为纯文本。”

We recommend using the `--json` argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json object per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset with nltk punctuation standardization. For BERT training, use the `--split-sentences` flag to `preprocess_data.py` as described [above](#data-preprocessing) to include sentence breaks in the produced index. If you'd like to use Wikipedia data for GPT training you should still clean it with nltk/spacy/ftfy, but do not use the `--split-sentences` flag.
我们建议在使用 WikiExtractor 时使用 --json 参数，这会将维基百科数据转储为松散的 json 格式（每行一个 json 对象），使其在文件系统中更易于管理，也更容易被我们的代码库使用。我们建议进一步使用 nltk 标点标准化对此 json 数据集进行预处理。对于 BERT 训练，请按照上述 描述使用 --split-sentences 标志将句子断句包含在生成的索引中。如果您想将维基百科数据用于 GPT 训练，则仍需使用 nltk/spacy/ftfy 进行清理，但不要使用 --split-sentences 标志。

## Collecting GPT Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filter, clean, and deduplicate all downloaded content according to the procedure described in our [openwebtext](./tools/openwebtext) directory. For reddit URLs corresponding to content up to October 2018 we arrived at approximately 37GB of content.
收集 GPT Webtext 数据
我们利用公开可用的 OpenWebText 库（来自 jcpeterson 和 eukaryote31 的工作）下载 URL。然后根据 openwebtext 目录中描述的流程过滤、清理和去重所有下载内容。对于截至 2018 年 10 月的 Reddit URL，我们获得了大约 37GB 的内容。

# Reproducibility
Megatron training can be bitwise reproducible; to enable this mode use `--deterministic-mode`. This means that the same training config run twice in the same HW and SW environment should produce identical model checkpoints, losses and accuracy metric values (iteration time metrics may vary).
可复现性
Megatron 训练可以实现位级复现；要启用此模式，请使用 --deterministic-mode。
这意味着在同一硬件和软件环境中 两次运行相同的训练配置 将生成相同的模型检查点、损失值和准确率指标值（迭代时间指标可能会有所不同）。

There are currently three known Megatron optimizations that break reproducibility whilst still producing almost identical training runs:
1. The specific NCCL algorithm that is used during an all-reduce (as specified by the environment variable `NCCL_ALGO`) is important. We have tested the following: `^NVLS`, `Tree`, `Ring`, `CollnetDirect`, `CollnetChain`. The code admits the use of `^NVLS`, which allows NCCL the choice of non-NVLS algorithms; its choice seems to be stable.
2. Flash attention is non-deterministic; do not use `--use-flash-attn`.
3. If using Transformer Engine, you must also set the environment variable `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`.
目前有三种已知的 Megatron 优化会破坏复现性，但仍会产生几乎相同的训练运行：
在 all-reduce 期间使用的特定 NCCL 算法（由环境变量 NCCL_ALGO 指定）很重要。我们测试了以下算法：^NVLS、Tree、Ring、CollnetDirect、CollnetChain。代码允许使用 ^NVLS，这使 NCCL 可以选择非 NVLS 算法；其选择似乎是稳定的。
Flash attention 是非确定性的；请勿使用 --use-flash-attn。
如果使用 Transformer Engine，则必须设置环境变量 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0。

In addition, determinisim has only been verified in NGC PyTorch containers up to and newer than 23.12. If you observe nondeterminism in Megatron training under other circumstances please open an issue.
此外，确定性仅在 NGC PyTorch 容器 23.12 及更新版本中得到验证。如果您在其他情况下观察到 Megatron 训练的非确定性，请提交问题。

# Checkpoint conversion 检查点转换

We support two forms of model conversion:

1. Model class conversion (i.e., the `GPTModel` in `model.legacy` vs. `model.core`)
2. Checkpoint format conversion (i.e., distributed vs. non-distributed checkpoint)
我们支持两种形式的模型转换：
模型类转换（例如，model.legacy 中的 GPTModel 与 model.core）
检查点格式转换（例如，分布式与非分布式检查点）

## Model class conversion 模型类转换

Megatron supports converting between different model classes, including internal model classes (we currently have the older `legacy` models, and the newer `core` models) and external model classes (such as Meta, Huggingface, Mistral, and Mixtral models). Additionally, during this conversion, one can update the parallel state of the model (i.e., changing tensor and pipeline model parallelism).
Megatron 支持在不同模型类之间进行转换，包括内部模型类（我们目前有较旧的 legacy 模型和较新的 core 模型）和外部模型类（如 Meta、Huggingface、Mistral 和 Mixtral 模型）。此外，在转换过程中，可以更新模型的并行状态（例如，更改张量和流水线模型并行性）。

 We provide the tool `tools/checkpoint/convert.py` to convert between model classes. Some important arguments include:
我们提供了工具 tools/checkpoint/convert.py 来在模型类之间进行转换。一些重要参数包括：

- `--model-type`: `GPT` or `BERT`
- `--loader`: format of the existing checkpoint. Supported formats include:
  - `legacy`: our older model classes (under `megatron.legacy.model`)
  - `core`: our newer model classes (under `megatron.core.models`)
  - `llama_mistral`: for loading Llama and Mistral models (supports Meta and Huggingface formats)
  - `mixtral_hf`: for loading Mixtral models (Huggingface only)
- `--load-dir`: directory for loading the existing checkpoint
- `--saver`: `legacy` or `core` (see descriptions under `--loader`)
- `--save-dir`: directory for saving the new checkpoint
- `--target-tensor-parallel-size`: new tensor model parallel size
- `--target-pipeline-parallel-size`: new pipeline model parallel size
--model-type：GPT 或 BERT
--loader：现有检查点的格式。支持的格式包括：
legacy：我们较旧的模型类（位于 megatron.legacy.model 下）
core：我们较新的模型类（位于 megatron.core.models 下）
llama_mistral：用于加载 Llama 和 Mistral 模型（支持 Meta 和 Huggingface 格式）
mixtral_hf：用于加载 Mixtral 模型（仅限 Huggingface）
--load-dir：加载现有检查点的目录
--saver：legacy 或 core（参见 --loader 下的描述）
--save-dir：保存新检查点的目录
--target-tensor-parallel-size：新的张量模型并行大小
--target-pipeline-parallel-size：新的流水线模型并行大小

For more argument details, please see the main script (`convert.py`), loader scripts (`loader_core.py`, `loader_legacy.py`, `loader_llama_mistral.py`, `loader_mixtral_hf.py`), or saver scripts (`saver_core.py`, `saver_legacy.py`).
更多参数详情，请参见主脚本（convert.py）、加载器脚本（loader_core.py、loader_legacy.py、loader_llama_mistral.py、loader_mixtral_hf.py）或保存器脚本（saver_core.py、saver_legacy.py）。

An example command for converting a GPT model from the old format (`legacy`) to the new format (`core`) would look as follows:
将 GPT 模型从旧格式（legacy）转换为新格式（core）的示例命令如下：

```
python tools/checkpoint/convert.py \
>   --model-type GPT \
>   --loader legacy \
>   --load-dir ${LEGACY_FORMAT_DIR} \
>   --saver core \
>   --save-dir ${CORE_FORMAT_DIR} \
>   --target-tensor-parallel-size ${TP} \
>   --target-pipeline-parallel-size ${PP} \
```

For examples of converting Llama/Mistral models into Megatron, please see [here](docs/llama_mistral.md).
有关将 Llama/Mistral 模型转换为 Megatron 的示例，请参见 此处。

## Checkpoint format conversion 检查点格式转换

Megatron offers multiple checkpoint formats, including:
Megatron 提供了多种检查点格式，包括：

- `torch`: Basic checkpoint format with sequential read & writes, and is tied to a specific tensor/pipeline model parallel state (TP/PP states, respectively). (While a specific checkpoint is tied to a specific TP/PP state, a checkpoint can still be manually converted via the model class converter described above).
- `torch_dist`: Distributed checkpoint format, for fast parallel reads & writes, and also is parallel state agnostic (i.e., one can load the same checkpoint to different TP/PP setups).
torch：基本检查点格式，顺序读写，并绑定到特定的张量/流水线模型并行状态（分别为 TP/PP 状态）。（虽然特定检查点绑定到特定的 TP/PP 状态，但仍然可以通过上述模型类转换器手动转换。）
torch_dist：分布式检查点格式，用于快速并行读写，并且与并行状态无关（即，可以将相同的检查点加载到不同的 TP/PP 设置中）。

Generally speaking, `torch_dist` is the more modern and recommended checkpoint format due to its speed. However, depending on the use case, it may be desirable to convert between these two formats. To do so, launch your *training* script (e.g., via `pretrain_gpt.py`) as you normally would, but with two additional arguments:
一般来说，torch_dist 是更现代且推荐的检查点格式，因为它速度更快。然而，根据用例，可能需要在这两种格式之间进行转换。为此，请像平常一样启动您的 训练 脚本（例如，通过 pretrain_gpt.py），但添加两个额外的参数：

- `--ckpt-convert-format ${FORMAT}`: `${FORMAT}` can be one of `torch` or `torch_dist`, as described above.
- `--ckpt-convert-save ${PATH_TO_SAVE_NEW_FORMAT}`: this path should be different than your existing `--load`/`--save` paths, to avoid overwriting the existing checkpoint. After converting, use this new path for your `--load`/`--save` paths.
--ckpt-convert-format ${FORMAT}：${FORMAT} 可以是 torch 或 torch_dist，如上所述。
--ckpt-convert-save ${PATH_TO_SAVE_NEW_FORMAT}：此路径应不同于现有的 --load/--save 路径，以避免覆盖现有检查点。转换后，使用此新路径作为您的 --load/--save 路径。

The general idea of this checkpoint format converter is that it launches the model just as one normally would for training, but before running any training iterations, it saves to the new checkpoint format, and then exits. It is important to note that all other launch args should remain the same, in order for the system to understand the previous checkpoint format.
此检查点格式转换器的基本思想是像平常一样启动模型进行训练，但在运行任何训练迭代之前，它会保存为新的检查点格式，然后退出。需要注意的是，所有其他启动参数应保持不变，以便系统能够理解之前的检查点格式。

# Projects Using Megatron
Below are some of the projects where we have directly used Megatron:
* [BERT and GPT Studies Using Megatron](https://arxiv.org/pdf/1909.08053.pdf)
* [BioMegatron: Larger Biomedical Domain Language Model](https://www.aclweb.org/anthology/2020.emnlp-main.379.pdf)
* [End-to-End Training of Neural Retrievers for Open-Domain Question Answering](https://arxiv.org/abs/2101.00408)
* [Large Scale Multi-Actor Generative Dialog Modeling](https://www.aclweb.org/anthology/2020.acl-main.8.pdf)
* [Local Knowledge Powered Conversational Agents](https://arxiv.org/abs/2010.10150)
* [MEGATRON-CNTRL: Controllable Story Generation with External Knowledge Using Large-Scale Language Models](https://www.aclweb.org/anthology/2020.emnlp-main.226.pdf)
* [RACE Reading Comprehension Dataset Leaderboard](http://www.qizhexie.com/data/RACE_leaderboard.html)
* [Training Question Answering Models From Synthetic Data](https://www.aclweb.org/anthology/2020.emnlp-main.468.pdf)
* [Few-shot Instruction Prompts for Pretrained Language Models to Detect Social Biases](https://arxiv.org/abs/2112.07868)
* [Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](https://arxiv.org/abs/2202.04173)
* [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)
* [Multi-Stage Prompting for Knowledgeable Dialogue Generation](https://arxiv.org/abs/2203.08745)
* [Evaluating Parameter Efficient Learning for Generation](https://aclanthology.org/2022.emnlp-main.319.pdf)
* [Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](https://arxiv.org/abs/2202.04173)
* [Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study](https://arxiv.org/abs/2304.06762)
* [InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining](https://arxiv.org/abs/2310.07713)
* [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)
使用 Megatron 的项目
以下是一些我们直接使用 Megatron 的项目：
使用 Megatron 的 BERT 和 GPT 研究
BioMegatron：更大规模的生物医学领域语言模型
端到端训练神经检索器用于开放域问答
大规模多角色生成对话建模
本地知识驱动的对话代理
MEGATRON-CNTRL：使用大型语言模型结合外部知识的可控故事生成
RACE 阅读理解数据集排行榜
从合成数据训练问答模型
少量指令提示检测预训练语言模型中的社会偏见
探索领域适应训练在净化大规模语言模型中的极限
使用 DeepSpeed 和 Megatron 训练 Megatron-Turing NLG 530B，一个大规模生成语言模型
多阶段提示生成知识对话
评估参数高效学习的生成能力
探索领域适应训练在净化大规模语言模型中的极限
我们是否应该使用检索预训练自回归语言模型？一项综合研究
InstructRetro：检索增强预训练后的指令微调
基于 Mamba 的语言模型的实证研究