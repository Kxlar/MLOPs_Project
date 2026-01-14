# MLOps project checklist

- [X] Create a git repository (M5)
- [X] Make sure that all team members have write access to the GitHub repository (M5)
- [X] Create a dedicated environment for you project to keep track of your packages (M2)
- [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
- [X] Fill out the data.py file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
- [X] Add a model to model.py and a training procedure to train.py and get that running (M6)
- [ ] Remember to either fill out the requirements.txt/requirements_dev.txt files or keeping your pyproject.toml/uv.lock up-to-date with whatever dependencies that you are using (M2+M6)
- [X] Remember to comply with good coding practices (pep8) while doing the project (M7)
- [X] Do a bit of code typing and remember to document essential parts of your code (M7)



# Ordered todos 
- [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
- [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)
- [ ] Use logging to log important events in your code (M14)


# @Lea : load_dinov3 hyperparameters in that 
- [ ] Write one or multiple configurations files for your experiments (M11)
- [ ] Use Hydra to load the configurations and manage your hyperparameters (M11)


# 
- [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)


- [ ] Construct one or multiple docker files for your code (M10)
- [ ] Build the docker files locally and make sure they work as intended (M10)
- [ ] Use profiling to optimize your code (M12)
- [ ] Consider running a hyperparameter optimization sweep (M14)


