# Triple CRISPR database construction pipeline
Implementation of Triple CRISPR database, first introduced in:

Mammalian Reverse Genetics without Crossing Reveals Nr3a as a Short-Sleeper Gene.
Sunagawa, et al. Cell Reports. (2016)
https://pubmed.ncbi.nlm.nih.gov/26774482/

# In a nutshell
This code allows users to design where to target on the genome of the organism of interest by SpCas9-based CRISPR system.
Targets will be searched through the specified genome (for each gene), and scored by their sequence, structure, and off-targets.
Output will be in the form of .csv files containing all information.

# Usage
It is assumed that your directory is structured as follows:

```
(Top directory)/
  ├ script/
  └ database/　
```

Upon execution, the script automatically generates a directory under `database`. All the subsequent outputs will be done solely inside here.

The interactive pipeline is in `script/pipeline_interactive.ipynb`.

# Environment

Necessary packages are listed in `scripts/python_requirement.txt`.

Alternatively, you can use a docker image which can be build with `scripts/dockerfile`.
