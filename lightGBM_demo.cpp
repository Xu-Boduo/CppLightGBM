#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/config.h>
#include <LightGBM/metric.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/objective_function.h>
#include <vector>
#include <fstream>
#include <filesystem>
#include <LightGBM/prediction_early_stop.h>
using namespace LightGBM;

Dataset* LoadDatasetFromVectorDemo()
{
    // 6 features, 5 samples, row is feature, col is sample
    std::vector<std::vector<double>> data_table = {
        {1.0, 1.1, 1.2, 1.3, 1.4},
        {2.0, 2.1, 2.2, 2.3, 2.4},
        {3.0, 3.1, 3.2, 3.3, 3.4},
        {4.0, 4.1, 4.2, 4.3, 4.4},
        {5.0, 5.1, 5.2, 5.4, 5.4},
        {6.0, 6.1, 6.2, 6.3, 6.4}
    };
    std::vector<float> labels = {0.0, 1.0, 2.0, 3.0, 4.0};

    // extend sample
    std::vector<std::vector<double>> extend_data(data_table.size());
    for (int k = 0; k < data_table.size(); k++) {
        extend_data[k].resize(data_table[k].size() *  100);
        labels.insert(labels.end(), labels.begin(), labels.end());
        for (int i = 0; i < 100; i++) {
            for (const auto &row : data_table) {
                extend_data[k].insert(extend_data[k].end(), row.begin(), row.end());
            }
        }
    }

    const auto num_samples = static_cast<int32_t>(extend_data[0].size());
    const int num_features = static_cast<int>(extend_data.size());
    // store the index of samples within each feature
    std::vector<std::vector<int>> sample_idx(num_features);
    for (int col = 0; col < num_features; ++col) {
        sample_idx[col].resize(num_samples);
        for (int row = 0; row < num_samples; ++row) {
            sample_idx[col][row] = row;
        }
    }

    Config config;
    config.max_bin = 16;
    config.force_col_wise = true;
    config.num_class = 5;
    DatasetLoader loader(config, nullptr, config.num_class, nullptr);
    Dataset* dataset = loader.ConstructFromSampleData(Common::Vector2Ptr<double>(&extend_data).data(),
        Common::Vector2Ptr<int>(&sample_idx).data(),
        num_features,
        Common::VectorSize<double>(extend_data).data(),
        num_samples,
        num_samples,
        num_samples);

    // set label
    dataset->SetFloatField("label", labels.data(), num_samples);
    return dataset;
}

void GenerateDataset(const std::string& filename) {
    // 5 samples, 6 features
    const std::vector<std::vector<float>> data = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        {1.1, 2.1, 3.1, 4.1, 5.1, 6.1},
        {1.2, 2.2, 3.2, 4.2, 5.2, 6.2},
        {1.3, 2.3, 3.3, 4.3, 5.3, 6.3},
        {1.4, 2.4, 3.4, 4.4, 5.4, 6.4}
    };
    const std::vector<float> labels = {0, 1, 0, 1, 0};

    std::vector<std::vector<float>> extend_data;
    std::vector<float> extend_labels;
    for (int i = 0; i < 100; i++) {
        for (const auto &row : data) {
            extend_data.emplace_back(row);
        }
        extend_labels.insert(extend_labels.end(), labels.begin(), labels.end());
    }

    // Save as File
    std::ofstream fOut(filename);
    for (size_t i = 0; i < extend_data.size(); ++i) {
        fOut << extend_labels[i] << ",";
        for (const float elem : extend_data[i]) {
            fOut << elem << ",";
        }
        fOut << "\n";
    }
    fOut.close();
}

std::vector<const Metric*> CovertMetricFormat(Config &config, Dataset* &dataset)
{
    std::vector<const Metric*> metrics;
    for (const auto &metric_type : config.metric) {
        Metric* metric = Metric::CreateMetric(metric_type, config);  // config is actually useless
        if (metric == nullptr) { continue; }
        metric->Init(dataset->metadata(), dataset->num_data());
        metrics.push_back(metric);
    }
    return metrics;
}

void ReadSampleToVector(const std::string& filename, std::vector<std::vector<double>> &features,
    std::vector<double> &labels) {
    // open file
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("could not open file：" + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> rowFeatures;
        std::istringstream lineSS(line);
        double value, label;
        lineSS >> label;
        while (lineSS >> value) {
            rowFeatures.push_back(value);
        }
        if (!rowFeatures.empty()) {
            labels.push_back(label);
            features.push_back(rowFeatures);
        }
    }
    file.close();
}

void LightGBMTrain()
{
    // generate train dataset
    // GenerateDataset(train_file);

    // load train dataset
    const std::string train_file = "../data/train.txt";
    Config config;
    config.label_column = "0";   // set label column, default 0
    config.max_bin = 255;
    config.force_col_wise = true;
    config.num_class = 5;
    DatasetLoader loader(config, nullptr, config.num_class, train_file.c_str());
    Dataset* train_data = loader.LoadFromFile(train_file.c_str());

    // load validation dataset
    const std::string valid_file = "../data/valid.txt";
    Dataset* valid_data = loader.LoadFromFileAlignWithOtherDataset(valid_file.c_str(), train_data);

    // create boosting
    config.boosting = "gbdt";
    config.input_model = "";
    Boosting* boosting = Boosting::CreateBoosting(config.boosting, config.input_model.c_str());

    // create objective function
    config.objective = "multiclass";
    ObjectiveFunction* objective_fun = ObjectiveFunction::CreateObjectiveFunction(config.objective, config);
    // initialize the objective function
    objective_fun->Init(train_data->metadata(), train_data->num_data());

    // set evaluation indicators
    config.metric = {"auc", "l1"};
    const std::vector<const Metric*> train_metrics = CovertMetricFormat(config, train_data);

    // set training parameters
    config.output_model = "../output/lightgbm.txt";
    config.snapshot_freq = -1;
    config.num_iterations = 10;  // 多分类问题保存的树的数量为 num_class * num_iterations

    // initialize the boosting
    boosting->Init(&config, train_data, objective_fun, train_metrics);
    const std::vector<const Metric*> valid_metrics = CovertMetricFormat(config, valid_data);
    boosting->AddValidDataset(valid_data, valid_metrics);

    // training
    boosting->Train(config.snapshot_freq, config.output_model);;
    boosting->SaveModelToFile(0, -1, 0, config.output_model.c_str());
}

void LightGBMPredict()
{
    Config config;
    config.input_model = "../data/pretrainModel.txt";
    config.boosting = "gbdt";

    Boosting* boosting = Boosting::CreateBoosting(config.boosting, config.input_model.c_str());
    config.predict_contrib = false;
    boosting->InitPredict(0, -1, config.predict_contrib);

    // load test data
    std::vector<std::vector<double>> features;
    std::vector<double> labels;
    const std::string testFile = "../data/test.txt";
    ReadSampleToVector(testFile, features, labels);

    std::vector<std::vector<double>> preds;
    preds.resize(features.size());
    config.pred_early_stop_margin = 10;
    config.pred_early_stop_freq = 2;
    PredictionEarlyStopConfig pred_early_stop_config{};
    pred_early_stop_config.margin_threshold = config.pred_early_stop_margin;
    pred_early_stop_config.round_period = config.pred_early_stop_freq;
    const PredictionEarlyStopInstance earlyStop =
        CreatePredictionEarlyStopInstance("multiclass", pred_early_stop_config);
    for (size_t i = 0; i < features.size(); ++i) {
        // 输出是经过 softmax 的概率结果, 还需进一步转为 label
        preds[i].resize( boosting->NumberOfClasses());
        boosting->Predict(features[i].data(), preds[i].data(), &earlyStop);
    }
}

int main()
{
    // LoadDatasetFromVectorDemo();
    LightGBMTrain();
    // LightGBMPredict();
    return 0;
}
