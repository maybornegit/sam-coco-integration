### 822-Project ~ SAM Integration w/ COCO

1. Clone SAM2 Repo and Download SAM2 Checkpoints (place inside checkpoints folder).
2. Move coco-label-papers.txt into the root directory, sam2 (should be in the same folder as checkpoints directory)
3. Make minor alterations to sam-dynamic-finder.py source code. Change string inside sys.path in line 8 and self.core_dir in line 32 to reflect where your SAM2 was downloaded
4. Install dependencies, preferably with conda
	- Create a new conda environment with 'conda create -n [name] python=3.11'
	- In the root directory of SAM2, run the command line 'pip install -e .'
	- Run 'conda install opencv matplotlib'
5. In the root directory of the file sam-dynamic-finder.py, run 'python sam-dynamic-finder.py' and observe the results. Out of the box it should continually run and output 1 or 0, as well as propagate video progress bars. If you have any questions about dependency issues, reach out.

