//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DistributedLearnerBase.h"
#include "Learner.h"

namespace CNTK
{
    DistributedLearnerBase::DistributedLearnerBase(DistributedCommunicatorPtr communicator, const std::vector<LearnerPtr>& learners, size_t distributeAfterSamples)
        : DistributedLearner(),
          m_communicator(communicator),
          m_learner(std::make_shared<CompositeLearner>(learners)),
          m_distributeAfterSamples(distributeAfterSamples),
          m_totalNumberOfSamplesSeen(0)
    {
    }

    // Get checkpoint state associated with distributed trainer
    Dictionary DistributedLearnerBase::CreateCheckpoint()
    {
        Dictionary result;
        result[L"localLearners"] = m_learner->Serialize();
        result[L"totalNumberOfSamplesSeen"] = m_totalNumberOfSamplesSeen;
        return result;
    }

    // Restores the state associated with distributed trainer
    void DistributedLearnerBase::RestoreFromCheckpoint(const Dictionary& checkpoint)
    {
        m_learner->RestoreFromCheckpoint(checkpoint[L"localLearners"].Value<Dictionary>());
        m_totalNumberOfSamplesSeen = checkpoint[L"totalNumberOfSamplesSeen"].Value<size_t>();
    }

    void DistributedLearnerBase::PrepaireZeroGradients(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& info)
    {
        // Need to intialize gradients to 0 in case when it is an empty minibatch.
        for (auto& g : gradientValues)
        {
            auto weights = g.first.Value();
            g.second = MakeSharedObject<NDArrayView>(0, weights->GetDataType(), weights->Shape(), weights->Device());
        }

        auto dataType = gradientValues.begin()->first.GetDataType();
        info.evalCriterionValue = MakeSharedObject<NDArrayView>(0, dataType, NDShape{ 1 }, DeviceDescriptor::CPUDevice());
        info.trainingLossValue = MakeSharedObject<NDArrayView>(0, dataType, NDShape{ 1 }, DeviceDescriptor::CPUDevice());
    }

    void DistributedLearnerBase::ConvertToOrdered(const std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, std::vector<std::pair<Parameter, NDArrayViewPtr>>& result)
    {
        result.reserve(gradientValues.size());
        result.clear();
        for (auto g : gradientValues)
            result.push_back(std::make_pair(g.first, g.second));

        std::sort(result.begin(), result.end(),
            [](const std::pair<Parameter, NDArrayViewPtr>& a, const std::pair<Parameter, NDArrayViewPtr>& b) { return a.first.Uid() < b.first.Uid(); });
    }

    const std::vector<LearnerPtr>& DistributedLearnerBase::ParameterLearners() const
    {
        return m_learner->ParameterLearners();
    }

    const std::vector<Parameter>& DistributedLearnerBase::Parameters() const
    {
        return m_learner->Parameters();
    }
}
