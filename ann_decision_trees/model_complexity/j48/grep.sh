#!/bin/bash
grep -r "Number of Leaves" | xargs sed -i 's/:/,/g' >> ../c48_ModelComplexity.csv