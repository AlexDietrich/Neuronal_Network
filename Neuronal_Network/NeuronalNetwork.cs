using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BIF4_MLE_NeuronalNetwork.utils;

namespace Neuronal_Network
{
    internal static class NeuronalNetwork
    {
        private static InputLayer InLayer { get; set; } = new InputLayer();
        private static HiddenLayer HidLayer { get; set; } = InLayer.ChildLayer;
        private static OutputLayer OutLayer { get; set; } = HidLayer.ChildLayer;

        public static HashSet<double> NetworkErrorRating { get; private set; } = new HashSet<double>();
        public static double TargetError { get; private set; } = 0.05;

        public static double LearningRate { get; private set; } = 0.2;
        public static double MomentumFactor { get; private set; } = 0.9;
        public static bool UseMomentum { get; private set; } = true;

        public static void Train()
        {
            var count = 0;
            while (CalculateNetworkErrorAverage() > TargetError)
            {
                NetworkErrorRating.Clear();
                foreach (var image in MnistReader.ReadTrainingData())
                {
                    count++;
                    if (count % 1000 == 0)
                        Console.WriteLine(
                            $"Aktueller Durchschnittlicher ErrorWert: {CalculateNetworkErrorAverage()} bei: {count} Durchläufen");
                    SetNeuronValueFromImage(image.Data); //Image data wird in InputLayer eingelesen 
                    SetDesiredValueFromLabel(image
                        .Label); //LabelDaten werden im OutputLayer gespeichert für späteren Vergleich
                    FeedForward();
                    BackProgagate();
                    NetworkErrorRating.Add(CalculateError());
                }
            }
        }

        public static void Test()
        {
            var gesamtElemente = 0;
            var korrekteElemente = 0;
            foreach (var image in MnistReader.ReadTestData())
            {
                gesamtElemente++;
                SetNeuronValueFromImage(image.Data); //Image data wird in InputLayer eingelesen 
                SetDesiredValueFromLabel(image.Label); //LabelDaten werden im OutputLayer gespeichert für späteren Vergleich
                FeedForward();
                for (var i = 0; i < OutLayer.DesiredValues.Length; i++)
                {
                    if (OutLayer.DesiredValues[i].CompareTo(1) != 0) continue;
                    if (OutLayer.GetHighestNeuronIndex() == i) korrekteElemente++;
                }
            }

            double accuracy = ((double)korrekteElemente * (double)100 / (double)gesamtElemente );
            Console.WriteLine($"Accuracy: {accuracy} %");
        }

        public static double CalculateNetworkErrorAverage()
        {
            if (NetworkErrorRating.Count == 0) return 0.09; //Für While Schleife...
            double average = 0.0;
            int count = 0;
            foreach (var error in NetworkErrorRating)
            {
                count ++;
                average += error;
            }
            average = average / count;
            return average;
        }

        /// <summary>
        /// Berechne Error Rate für eine Iteration vom Neuronalen Netzwerk.
        /// </summary>
        /// <returns></returns>
        public static double CalculateError()
        {
            double error = 0.0;
            for (var i = 0; i < OutLayer.NeuronValue.Length; i++)
            {
                error += Math.Pow(OutLayer.NeuronValue[i] - OutLayer.DesiredValues[i], 2);
            }
            error = error / OutLayer.NeuronValue.Length;
            return error;
        }

        private static void SetDesiredValueFromLabel(double imageLabel)
        {
            ResetDesiredValue();
            for (var i = 0; i < OutLayer.DesiredValuesBlueprint.Length; i++)
            {
                if (OutLayer.DesiredValuesBlueprint[i].CompareTo(imageLabel) != 0) continue;
                OutLayer.DesiredValues[i] = 1.0; //Die Stelle welche der Zahl entspricht setze auf 1 -> alle anderen bleiben auf 0
                break;
            }
        }

        private static void ResetDesiredValue()
        {
            for (var i = 0; i < OutLayer.DesiredValues.Length; i++)
            {
                OutLayer.DesiredValues[i] = 0.0; //Die Stelle welche der Zahl entspricht setze auf 1 -> alle anderen bleiben auf 0
            }
        }

        private static void SetNeuronValueFromImage(double[,] data)
        {
            for (var i = 0; i < data.GetLength(0); i++)
            {
                for (var j = 0; j < data.GetLength(1); j++)
                {
                    InLayer.NeuronValue[i * 28 + j] = data[i, j]; //Splitte zweidimensionales Array in eindimensionales auf und speicher daten im InputLayer
                }
            }
        }

        public static void FeedForward()
        {
            InLayer.CalculateNeuronValues();
            HidLayer.CalculateNeuronValues();
            OutLayer.CalculateNeuronValues();
        }

        public static void BackProgagate()
        {
            OutLayer.CalculateErrors();
            HidLayer.CalculateErrors();
            HidLayer.AdjustWeights();
            InLayer.AdjustWeights();
        }
    }
}
