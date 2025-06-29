#!/bin/bash

for bubble_name in */; do
	cd "$bubble_name"
	bubble_name="${bubble_name%/}"
	python ../../find_bubbles.py "$bubble_name".pdb
	python ../../find_bubbles_cuda.py "$bubble_name".pdb
	#mv bubbles.pdb bubbles_0.7_cut.pdb
	cd ..
done
