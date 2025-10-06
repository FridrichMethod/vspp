#!/bin/bash

for i in $(seq 0 15); do if grep -q "WARNING Unable to checkout 8 PSP_PLOP v69 license(s). Error is -4: Licensed number of users already reached. Tried 3 time(s) in 30 seconds." "lig_${i}.log"; then echo -n "$i,"; fi; done; echo ""