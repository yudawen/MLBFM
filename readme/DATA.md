# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation and training, you will need to setup dataset.


### BUFF dataset

- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${BUFF_ROOT}
  |-- train_data
  `-- |-- train
      `-- |-- image
          |-- gt_txt
  `-- |-- val
    `-- |-- image
        |-- gt_txt

  ~~~
An example of a annotation file (xxx.txt):

1 462 506 459 511 472 511 464 506
1 327 503 326 511 332 511 331 503
1 284 503 279 505 281 511 287 511
1 257 503 237 511 262 510 260 504
1 359 506 361 511 373 511 370 502
1 85 501 84 511 101 511 101 502
...

format: category_id (starting  from 1) x1 y1 x2 y2 x3 y3 .....

