﻿using System;

namespace Neuronal_Network
{
    internal class HiddenLayer
    {
        /// <summary>
        /// Initialize all needed variables for the neural network. 28x28 Pixels/Picture = 784
        /// </summary>
        private const int NumberOfNeurons = 89;
        private const int NumberOfParentNeurons = 784;
        private const int NumberOfChildNeurons = 10;

        public double[] NeuronValue { get; set; } = new double[NumberOfNeurons];
        public double[] Weight { get; set; } = new double[NumberOfNeurons*NumberOfNeurons];
        private double[] WeightChanges { get; set; } = new double[NumberOfNeurons*NumberOfNeurons];
        public double[] Bias { get; set; } = new double[NumberOfNeurons];
        public double[] BiasWeight { get; set; } = new double[NumberOfNeurons];
        public double[] Error { get; set; } = new double[NumberOfNeurons];


        private readonly bool LinearOutput = false; //Kein Linearer Output erwünscht :D 

        public OutputLayer ChildLayer { get; private set; }
        private InputLayer ParentLayer { get; set; }

        public HiddenLayer(InputLayer parent)
        {
            ParentLayer = parent;
            ChildLayer = new OutputLayer(this);
            FillWeightWithRandomValues();
        }

        private void FillWeightWithRandomValues()
        {
            var rnd = new Random();
            for (var i = 0; i < NumberOfNeurons; i++)
            {
                BiasWeight[i] = rnd.NextDouble() * 2 - 1;
                Bias[i] = (rnd.NextDouble() < 0.5) ? 1 : -1;
                for (var j = 0; j < NumberOfNeurons; j++)
                {
                    Weight[i*NumberOfNeurons+j] = rnd.NextDouble() * 2 - 1;
                    WeightChanges[i*NumberOfNeurons+j] = 0.0;
                }
            }
        }

        public void CalculateNeuronValues()
        {
            double x = 0.0;
            if (ParentLayer == null) return;
            for (var j = 0; j < NumberOfNeurons; j++)
            {
                x = 0.0;
                for (var i = 0; i < NumberOfParentNeurons; i++)
                {
                    x += ParentLayer.NeuronValue[i] * ParentLayer.Weight[i*NumberOfParentNeurons+j];
                }
                x += ParentLayer.Bias[j] * ParentLayer.BiasWeight[j];
                if ((ChildLayer == null) && LinearOutput)
                {
                    NeuronValue[j] = x;
                }
                else
                {
                    double k = Math.Exp(x);
                    NeuronValue[j] = k / (1.0 + k);
                }
            }
        }

        public void CalculateErrors()
        {
            double sum = 0.0;
            for (var i = 0; i < NumberOfNeurons; i++)
            {
                sum = 0.0;
                for (var j = 0; j < NumberOfChildNeurons; j++)
                {
                    sum += ChildLayer.Error[j] * Weight[i*NumberOfNeurons+j];
                }
                Error[i] = sum * NeuronValue[i] * (1.0 - NeuronValue[i]);
            }

        }

        public void AdjustWeights()
        {
            double dw = 0.0;
            if (ChildLayer == null) return;
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                for (int j = 0; j < NumberOfChildNeurons; j++)
                {
                    dw = NeuronalNetwork.LearningRate * ChildLayer.Error[j] * NeuronValue[i];
                    if (NeuronalNetwork.UseMomentum)
                    {
                        Weight[i*NumberOfNeurons+j] += dw + NeuronalNetwork.MomentumFactor * WeightChanges[i*NumberOfNeurons+j];
                        WeightChanges[i*NumberOfNeurons+j] = dw;
                    }
                    else
                    {
                        Weight[i*NumberOfNeurons+j] += dw;
                    }
                }
            }
            for (int j = 0; j < NumberOfChildNeurons; j++)
            {
                BiasWeight[j] += NeuronalNetwork.LearningRate * ChildLayer.Error[j] * Bias[j];
            }
        }
    }
}
