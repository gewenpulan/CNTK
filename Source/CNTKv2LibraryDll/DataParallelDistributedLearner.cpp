//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataParallelDistributedLearner.h"
#include "DistributedCommunicator.h"
#include "Learner.h"

#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
#include "QuantizedDistributedCommunicator.h"
#include "QuantizedDataParallelDistributedLearner.h"
#include "BlockMomentumDistributedLearner.h"
#endif

namespace CNTK
{
#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits)
    {
        return MakeSharedObject<QuantizedMPICommunicatorImpl>(zeroThresholdFor1Bit, useQuantizationForSelfStripe, numQuantizationBits);
    }

    DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(
        QuantizedDistributedCommunicatorPtr communicator,
        const std::vector<LearnerPtr>& learners,
        size_t distributeAfterSamples,
        bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<QuantizedDataParallelDistributedLearner>(communicator, learners, distributeAfterSamples, useAsyncBufferedParameterUpdate);
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        const std::vector<LearnerPtr>& learners,
        size_t distributeAfterSamples,
        size_t blockSize,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedLearner>(
            communicator,
            learners,
            distributeAfterSamples,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate);
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        const std::vector<LearnerPtr>& learners,
        size_t distributeAfterSamples,
        size_t blockSize,
        double blockMomentumAsTimeConstant,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedLearner>(
            communicator,
            learners,
            distributeAfterSamples,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate,
            blockMomentumAsTimeConstant);
    }

#else
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool, bool, size_t)
    {
        LogicError("Quantized MPI Communicator is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr, const std::vector<LearnerPtr>&, size_t, bool)
    {
        LogicError("Quantized Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr /*communicator*/,
        const std::vector<LearnerPtr>&,
        size_t /*distributeAfterSamples*/,
        size_t /*blockSize*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr /*communicator*/,
        const std::vector<LearnerPtr>&,
        size_t /*distributeAfterSamples*/,
        size_t /*blockSize*/,
        double /*blockMomentumAsTimeConstant*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }
#endif

    DistributedLearnerPtr CreateDataParallelDistributedLearner(DistributedCommunicatorPtr communicator, const std::vector<LearnerPtr>& learners, size_t distributedAfterSamples, bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<DataParallelDistributedLearner>(communicator, learners, distributedAfterSamples, useAsyncBufferedParameterUpdate);
    }

    DataParallelDistributedLearner::DataParallelDistributedLearner(DistributedCommunicatorPtr communicator, const std::vector<LearnerPtr>& learners, size_t distributedAfterSamples, bool useAsyncBufferedParameterUpdate)
        : DistributedLearnerBase(communicator, learners, distributedAfterSamples)
    {
        if (useAsyncBufferedParameterUpdate)
            LogicError("Asynchronous parameter update is not yet supported.");
    }

    bool DataParallelDistributedLearner::Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& info, size_t& totalNumberOfSampleSeen)
    {
        if (m_totalNumberOfSamplesSeen >= m_distributeAfterSamples)
        {
            if (info.IsEmpty())
                PrepaireZeroGradients(gradientValues, info);
            ConvertToOrdered(gradientValues, m_gradientBuffer);

            std::vector<NDArrayViewPtr> valuesToAggregate;
            for (const auto& i : m_gradientBuffer)
                valuesToAggregate.push_back(i.second);
            valuesToAggregate.push_back(info.evalCriterionValue);
            valuesToAggregate.push_back(info.trainingLossValue);

            auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{ 1 }, DeviceDescriptor::CPUDevice());
            valuesToAggregate.push_back(value);

            m_communicator->AggregateInPlace(valuesToAggregate, m_communicator->Workers());
            info.numberOfSamples = static_cast<size_t>(*valuesToAggregate.back()->WritableDataBuffer<double>());
        }

        m_totalNumberOfSamplesSeen += info.numberOfSamples;
        totalNumberOfSampleSeen = m_totalNumberOfSamplesSeen;
        m_gradientBuffer.clear();

        auto result = m_learner->Update(gradientValues, info.numberOfSamples);
        return result && !info.IsEmpty();
    }
}
