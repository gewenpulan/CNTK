﻿# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import math
import numpy as np
import pytest
from .. import Function
from ..trainer import *
from ..learner import *
from .. import distributed
from .. import cross_entropy_with_softmax, classification_error, parameter, \
        input_variable, times, plus, reduce_sum

def create_data_parallel_distributed_learner(learner, quantized):
    return distributed.data_parallel_distributed_learner(
        learner=learner,
        use_async_buffered_parameter_update=False,
        num_quantization_bits=(1 if quantized else 32))

def create_block_momentum_distributed_learner(learner):
    return distributed.block_momentum_distributed_learner(
        learner=learner,
        block_size=1024)

def create_block_momentum_distributed_learner_with_time_constant(learner):
    return distributed.block_momentum_distributed_learner(
        learner=learner,
        block_size=1024,
        block_momentum_as_time_constant=4096)

def run_distributed_training(tmpdir, create_func):

    in1 = input_variable(shape=1)
    labels = input_variable(shape=1)
    p = parameter(shape=2, init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    dist_learner = create_func(momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant))

    communicator = dist_learner.communicator()
    workers = communicator.workers()
    current_worker = communicator.current_worker()
    found_rank = False
    for wk in workers:
        if current_worker.global_rank == wk.global_rank:
            found_rank = True

    assert found_rank

    trainer = Trainer(z, ce, errs, [ dist_learner ])
    in1_value = [[1],[2]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])
    
    p = str(tmpdir / 'checkpoint.dat')
    trainer.save_checkpoint(p)
    trainer.restore_from_checkpoint(p)

    communicator.barrier()
    
    assert trainer.model.name == 'z'

    # Ensure that Swig is not leaking raw types
    assert isinstance(trainer.model, Function)
    assert trainer.model.__doc__

def test_distributed(tmpdir, is_1bit_sgd):
    quantized=(True if is_1bit_sgd==1 else False)

    simple_aggregation=lambda learner: create_data_parallel_distributed_learner(learner, False)
    run_distributed_training(tmpdir, create_func=simple_aggregation)

    if is_1bit_sgd == 1:
        quantized_aggregation=lambda learner: create_data_parallel_distributed_learner(learner, True)
        run_distributed_training(tmpdir, create_func=quantized_aggregation)

        run_distributed_training(tmpdir, create_func=create_block_momentum_distributed_learner)
        run_distributed_training(tmpdir, create_func=create_block_momentum_distributed_learner_with_time_constant)
    distributed.Communicator.finalize()
