//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"

namespace CNTK
{
    ///
    /// Base class for distributed learners.
    ///
    class DistributedLearnerBase : public DistributedLearner
    {
    public:
        Dictionary CreateCheckpoint() override;

        void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

        DistributedCommunicatorPtr GetCommunicator() override
        {
            return m_communicator;
        }

        const std::vector<Parameter>& Parameters() const override;
        const std::vector<LearnerPtr>& ParameterLearners() const override;

    protected:
        DistributedLearnerBase(DistributedCommunicatorPtr communicator, const std::vector<LearnerPtr>& learners, size_t distributeAfterSamples);

        static void PrepaireZeroGradients(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& info);
        static void ConvertToOrdered(const std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, std::vector<std::pair<Parameter, NDArrayViewPtr>>& result);

        const DistributedCommunicatorPtr m_communicator;
        const CompositeLearnerPtr m_learner;
        const size_t m_distributeAfterSamples;

        size_t m_totalNumberOfSamplesSeen;
        std::vector<std::pair<Parameter, NDArrayViewPtr>> m_gradientBuffer;
        std::vector<Parameter> m_parameters;

        DistributedLearnerBase(const DistributedLearnerBase&) = delete;
        DistributedLearnerBase& operator=(const DistributedLearnerBase&) = delete;
        DistributedLearnerBase& operator=(DistributedLearnerBase&&) = delete;
        DistributedLearnerBase(DistributedLearnerBase&& other) = delete;
    };
}