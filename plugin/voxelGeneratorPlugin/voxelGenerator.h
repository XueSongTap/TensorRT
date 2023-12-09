/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_VOXEL_GENERATOR_H
#define TRT_VOXEL_GENERATOR_H

#include "NvInferPlugin.h"
#include "common/bboxUtils.h"
#include "common/kernels/kernel.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
// 这个头文件定义了一个名为 VoxelGeneratorPlugin 的 TensorRT 插件类，它是用来生成体素（voxels）的，通常用于三维点云数据处理。
// 这个插件类继承自 nvinfer1::IPluginV2DynamicExt，这是所有 TensorRT 动态扩展插件必须继承的基类。


namespace nvinfer1
{
namespace plugin
{

class VoxelGeneratorPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    //构造函数1 (VoxelGeneratorPlugin(int32_t maxVoxels, ...))：第一个构造函数接受体素生成器的配置参数，如最大体素数量、最大点数、体素特征数量和体素生成的空间范围。
    
    // 构造函数2 (VoxelGeneratorPlugin(int32_t maxVoxels, ..., int32_t gridZ))：第二个构造函数相比于第一个构造函数，包含了更多的参数，包括点云特征数量和网格尺寸，这些可以用来配置体素空间的分辨率。

    //构造函数3 (VoxelGeneratorPlugin(void const* data, size_t length))：一个构造函数用于反序列化操作，它使用一个指向缓冲区的指针和长度来初始化插件的状态。
    VoxelGeneratorPlugin() = delete;
    VoxelGeneratorPlugin(int32_t maxVoxels, int32_t maxPoints, int32_t voxelFeatures, float xMin, float xMax, float yMin,
        float yMax, float zMin, float zMax, float pillarX, float pillarY, float pillarZ);
    VoxelGeneratorPlugin(int32_t maxVoxels, int32_t maxPoints, int32_t voxelFeatures, float xMin, float xMax, float yMin,
        float yMax, float zMin, float zMax, float pillarX, float pillarY, float pillarZ, int32_t pointFeatures,
        int32_t gridX, int32_t gridY, int32_t gridZ);
    VoxelGeneratorPlugin(void const* data, size_t length);
    // IPluginV2DynamicExt Methods
    //克隆方法 (clone())：创建这个插件的深拷贝。

    // 获取输出维度 (getOutputDimensions(...))：给定输入的维度，返回输出的维度。

    // 支持的格式组合 (supportsFormatCombination(...))：判断插件是否支持某种特定的输入/输出数据格式组合。

    // 配置插件 (configurePlugin(...))：在推理开始前，根据输入和输出的配置调整插件的状态。

    // 获取工作空间大小 (getWorkspaceSize(...))：返回执行这个插件所需的额外工作空间大小。

    // 执行 (enqueue(...))：执行插件的前向计算，它是在推理时被调用的函数。
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    //VoxelGeneratorPlugin 类还包括多个私有成员变量，用于存储插件的配置和状态信息，例如：

    // mPillarNum：体素的数量。
    // mPointNum：每个体素中的点数。
    // mFeatureNum：体素的特征数量。
    // mMinXRange、mMaxXRange、mMinYRange 等：定义体素化采样空间的参数。
    // mPillarXSize、mPillarYSize、mPillarZSize：体素的大小。
    // mPointFeatureNum：点云中每个点的特征数量。
    // mGridXSize、mGridYSize、mGridZSize：体素网格的大小。
    // 此外，还提供了插件的命名空间设置和序列化方法。插件创建者类还有一些用于管理和注册插件的静态成员变量。
    std::string mNamespace;
    // Shape Num for *input*
    int32_t mPillarNum;
    int32_t mPointNum;
    int32_t mFeatureNum;
    float mMinXRange;
    float mMaxXRange;
    float mMinYRange;
    float mMaxYRange;
    float mMinZRange;
    float mMaxZRange;
    float mPillarXSize;
    float mPillarYSize;
    float mPillarZSize;
    // feature number of pointcloud points: 4 or 5
    int32_t mPointFeatureNum;
    int32_t mGridXSize;
    int32_t mGridYSize;
    int32_t mGridZSize;
};

// 插件创建类 VoxelGeneratorPluginCreator
// 构造函数 (VoxelGeneratorPluginCreator())：初始化插件创建者。

// 获取插件名称 (getPluginName())：返回插件的名称。

// 获取插件版本 (getPluginVersion())：返回插件的版本。

// 获取字段名称 (getFieldNames())：返回一个包含可用于插件构造的字段的集合。

// 创建插件 (createPlugin(...))：根据传入的 PluginFieldCollection 创建一个插件实例。

// 反序列化插件 (deserializePlugin(...))：从序列化的数据中创建一个插件实例。
class VoxelGeneratorPluginCreator : public nvinfer1::IPluginCreator
{
public:
    VoxelGeneratorPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
