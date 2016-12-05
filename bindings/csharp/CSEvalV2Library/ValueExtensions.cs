﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace CNTK
{
    public static class ValueExtensions
    {
        // The value represents a n-dimensional tensor with 2 dynamic axes: sequence and batch
        // It assumes that only the highest 2 axes are dynamic, and all the other axes are static. 
        public static void CopyTo<T>(this Value value, List<List<T>> data)
        {

            if ((value.GetDataType() == DataType.Float) && (!typeof(T).Equals(typeof(float))) || 
                (value.GetDataType() == DataType.Double) && (!typeof(T).Equals(typeof(double))))
            {
                throw new ArgumentException("The value type does not match the list type.");
            }

            // Todo: transform sparse to dense
            // Currently only for dense
            if ((value.GetStorageFormat() != StorageFormat.Dense))
            {
                throw new ArgumentException("The value is not in denst format.");
            }

            var outputNDArrayView = value.Data();
            var outputShape = outputNDArrayView.Shape();
            var outputShapeRank = outputShape.Rank();
            var variableShape = outputShape.SubShape(0, outputShapeRank - 2);

            Debug.Assert(outputShape.Rank() >= 3);
            var numOfElementsInSample = variableShape.TotalSize();
            var numOfSamplesInSequence = outputShape.GetDimensionSize(outputShapeRank - 2);
            var numOfSequences = outputShape.GetDimensionSize(outputShapeRank - 1);

            // Copy the data from the output buffer.
            // Todo: directly access the data in output buffer?
            // Todo: need to map DataBuffer() to C#
            NDArrayView cpuOutputNDArrayView;
            uint numOfOutputData = outputNDArrayView.Shape().TotalSize();
            // Todo: consider mask.
            Debug.Assert(numOfElementsInSample * numOfSamplesInSequence * numOfSequences == numOfOutputData);
            T[] outputData = new T[numOfOutputData];
            if (value.GetDataType() == DataType.Float)
            {
                cpuOutputNDArrayView = new NDArrayView(outputNDArrayView.Shape(), outputData as float[], numOfOutputData, DeviceDescriptor.CPUDevice());
            }
            else if (value.GetDataType() == DataType.Double)
            {
                cpuOutputNDArrayView = new NDArrayView(outputNDArrayView.Shape(), outputData as double[], numOfOutputData, DeviceDescriptor.CPUDevice());
            }
            else
            {
                throw new ArgumentException("The data type " + value.GetDataType().ToString() + " is not supported. Only float or double is supported by CNTK.");
            }

            cpuOutputNDArrayView.CopyFrom(outputNDArrayView);
            for (int seqIndex = 0, dataIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                var seqData = new List<T>();
                // Todo: consider mask
                // Todo: make it more efficient.
                for (int i = 0; i < numOfElementsInSample * numOfSamplesInSequence; i++)
                {
                    seqData.Add(outputData[dataIndex++]);
                }
                data.Add(seqData);
            }
        }

        // The value represents a n-dimensional tensor with 2 dynamic axes: sequence and batch
        public static void CopyTo<T>(this Value value, List<List<long>> data)
        {
            throw new NotImplementedException("Not implemented");
        }

        // The value represents a n-dimensional tensor with 2 dynamic axes: sequence and batch
        public static void CopyTo<T>(List<List<T>> data, List<List<long>> indexes, List<List<long>> nnzCounts)
        {
            throw new NotImplementedException("Not implemented");
        }

    }
}
