﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuronal_Network
{
    internal class InputLayer : Layer
    {
        /// <summary>
        /// Initialize all needed variables for the neural network. 28x28 Pixels/Picture = 784
        /// </summary>
        public const int NumberOfNeurons = 784;
        public const int NumberOfChildNeurons = 89;
        public const int NumberOfParentNeurons = 0;

        public double[] NeuronValue { get; set; } = new double[NumberOfNeurons];
        public double[,] Weight { get; set; } = new double[NumberOfNeurons, NumberOfNeurons];
        public double[,] WeightChanges { get; set; } = new double[NumberOfNeurons, NumberOfNeurons];
        public double[] Bias { get; set; } = new double[NumberOfNeurons];
        public double[] BiasWeight { get; set; } = new double[NumberOfNeurons];
        public bool LinearOutput = false; //Kein Linearer Output erwünscht :D 
        public double[] Error { get; set; } = new double[NumberOfNeurons];

        public HiddenLayer ChildLayer { get; private set; }
        public InputLayer ParentLayer { get; private set; } = null; //InputLayer doesn't have parents -> That makes me sad :'(

        public InputLayer()
        {
            FillWeightWithRandomValues();
            ChildLayer = new HiddenLayer(this);
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
                    Weight[i, j] = rnd.NextDouble() * 2 - 1;
                    WeightChanges[i, j] = 0.0;
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
                    x += ParentLayer.NeuronValue[i] * ParentLayer.Weight[i, j];
                }
                x += ParentLayer.Bias[j] * ParentLayer.BiasWeight[j];
                if ((ChildLayer == null) && LinearOutput)
                {
                    NeuronValue[j] = x;
                }
                else
                {
                    NeuronValue[j] = 1.0 / (1.0 + Math.Exp(-x));
                }
            }
        }

        public void AdjustWeights()
        {
            double dw = 0.0;
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                for (int j = 0; j < NumberOfChildNeurons; j++)
                {
                    dw = NeuronalNetwork.LearningRate * ChildLayer.Error[j] * NeuronValue[i];
                    if (NeuronalNetwork.UseMomentum)
                    {
                        Weight[i, j] += dw + NeuronalNetwork.MomentumFactor * WeightChanges[i, j];
                        WeightChanges[i, j] = dw;
                    }
                    else
                    {
                        Weight[i, j] += dw;
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
