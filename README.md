# CAP_Benchmark

This repository is providing the code and datasets to our paper **Reasoning with Preference Constraints: A Benchmark for Language Models in Many-to-One Matching Markets**, which provide 369 instances of the College Admission Problem, as well as the code to produce instances with varying parameters and LLM generation of those. 

## Install
Our code is made to be reproducible and extendable. During our experiments, we realize that one of the model tested, GPT-oss 120B, was particularly sensible to quantization and environmental setup. 

Therefore, we recommend creating the environment using `pip install -r requirements.txt`. 

## Instances - Dataset
The 369 instances are provided under `dataset_instance`. The solution of each of them using the DA algorithm is respectively provided in `dataset_match`. 
While our instances were specifically created to test varying parameters, such as number of students, number of schools, the capacity of the schools and the type of preferences, the file `creation_instances.jl` allow to generate new instances, which could be useful to extend to broader parameters, or simply generate new instances with new seeds. 

To run the file, you can just use julia `create_instances.jl`, or you can pass your arguments as `julia create_instances.jl --num-students 5,10,15,20 --preferences Complete --seeds 1,2,3 --input-dir my_instances --match-dir my_matches`. Note that for each number of students, particular number of schools were decided based on ratios and accordingly total capacity. As they are different per students, they remain as a constant that needs to be changed in the file. 


## LLM generation

### Iterative Prompting
