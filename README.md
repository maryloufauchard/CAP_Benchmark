# College Admission Benchmark for LLMs

This repository provides the code and datasets accompanying our paper **"Reasoning with Preference Constraints: A Benchmark for Language Models in Many-to-One Matching Markets."** It includes 369 instances of the College Admissions Problem, along with code to generate new instances with varying parameters and to run LLM generation on them.

## Installation
Our code is designed to be reproducible and extensible. During our experiments, we found that one of the tested models, GPT-oss-120B, was particularly sensitive to quantization and environment configuration.

We therefore recommend creating a dedicated environment and installing dependencies via: `pip install -r requirements.txt`. 

## Instances - Dataset
The 369 instances are provided under `dataset_instance`. with the corresponding Deferred Acceptance (DA) solution for each provided under `dataset_match`. 

While our instances were constructed to test a specific set of varying parameters, such as number of students, number of schools, the capacity of the schools and the type of preferences, the file `creation_instances.jl` can be used to generate new instances, e.g. to extend the parameter range or to generate fresh instances with new seeds to prevent memorization.

The script can be run with default parameters: `julia create_instances.jl`, or you can pass your arguments as 
```python
julia create_instances.jl --num-students 5,10,15,20 --preferences Complete --seeds 1,2,3 --input-dir my_instances --match-dir my_matches
```

Note: for each student count, the corresponding number of schools is chosen based on fixed ratios and total capacities. Since these mappings differ per student count, they are kept as constants within the file and must be edited directly if a different configuration is needed.

## LLM generation
With the generated instances, LLM generation can be run via `generation_scp.py`. The prompt instruction and other generation parameters are set through a config file (see the example in `config.py`). The corresponding prompt templates are available under `Prompt`. 
The generation can be done by calling it as such: 
`python generation_scp.py --model model_name --config_file path_to_config`. 

For all instances, one file with the generated answers will be created. The file `extract_match.py` is then used to extract the matching from each generated file, with tolerance for variation in output structure. Both the input folder (containing generated answers) and the output folder must be provided:
`python extract_matching.py --input path_input --output path_output`

Next, `metric_scp.py` computes the four evaluation metrics: feasibility, assignment stability, matching stability and (student) optimality. It is compared with the actual DA solutions: 

```python
python compute_metrics.py \  
  --instance-dir /dataset_instance \  
  --llm-match-dir extract_match_path \  
  --real-match-dir /dataset_match \  
  --output-dir output_path
```

Finally, `aggregation_metric.py` aggregates the per-instance metrics into a single summary, averaged overall and broken down by number of students and preference type:
```python
python aggregation_metric.py  \
  --metric-folder metric_scp_path \
  --output-folder output_path_folder \
  --output-filename output_path_file \
  --type-str basic
```

`--type_str` identifies the prompting strategy used to generate the evaluated outputs, and takes one of the following values: basic, basic_role, CoT_pseudo, CoT_python, CoT_txt, CoT_unsupervised, ICL_1, ICL_steps, vague for the prompt instruction Basic, Role, CoT pseudo code, CoT python, CoT text, CoT unsupervised, ICL, ICL w steps and General. 

### Iterative Prompting
For iterative prompting, where automatic feedback is computed on a proposed solution and the model is asked to revise it (up to a maximum number of attempts), the corresponding generation script is generation_iterative.py. Extraction and evaluation follow the same procedure described above.

Since we found that improvement across iterations was not always monotonic, the first, best, and last matchings produced for each instance are each saved to their own dedicated output folder.

### Citations 
```
@article{CAPbenchmark2026fauchard,
  title   = {Reasoning with Preference Constraints: A Benchmark for Language Models in Many-to-One Matching Markets},
  author  = {Marylou Fauchard and Florian Carichon and Margarida Carvalho and Golnoosh Farnadi},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = { https: // openreview. net/ forum? id= 2dpt2Ughzt}
}
```
