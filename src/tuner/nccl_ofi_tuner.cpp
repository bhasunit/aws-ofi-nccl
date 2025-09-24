/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cassert>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nccl/tuner.h>

#include "internal/tuner/nccl_defaults.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_system.h"
#include "nccl_ofi_param.h"

#include "tuner/nccl_ofi_tuner_region.h"
#include "tuner/nccl_ofi_tuner_model.h"
#include "tuner/nccl_ofi_tuner.h"

pthread_mutex_t nccl_ofi_tuner_ctx_lock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t nccl_ofi_tuner_destroy(void *context)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (ctx != NULL) {
		if (ctx->destroy_internal != NULL) {
			ret = ctx->destroy_internal(ctx);
		}
		free(ctx);
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ret;
}

static ncclResult_t nccl_ofi_tuner_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
{
	const char *platform_type = NULL;
	ncclResult_t ret = ncclSuccess;
	*context = NULL;
	nccl_ofi_tuner_context_t *ctx = NULL;
	bool region_support, model_support;
	int is_force_type_model = 0;
	enum nccl_ofi_tuner_platform tuner_platform;

	if (ofi_log_function == NULL) {
		ofi_log_function = logFunction;
	}

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);

	/*
	 * Retrieve platform type and pass to Region and Model based tuner support check functions.
	 * If both Region and Model based tuner are not supported, log a warning and exit.
	 */
	platform_type = nccl_net_ofi_get_product_name();
	if (platform_type == NULL) {
		NCCL_OFI_WARN("NCCL_OFI_TUNER is not available because platform type is unavailable.");
		goto exit;
	}

	if (ofi_nccl_tuner_force_type.get() == TUNER_TYPE::INTERNAL) {
		/* fallback to NCCL internal tuner */
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
			      "NCCL_OFI_TUNER_TYPE is Internal, Fall back to NCCL's tuner for platform : %s",
			      platform_type);
		goto exit;
	} else if (ofi_nccl_tuner_force_type.get() == TUNER_TYPE::MODEL) {
		is_force_type_model = 1;
	}

	if (ofi_nccl_force_num_rails.get_source() != ParamSource::DEFAULT) {
		// Because the tuner init is a local call, there is not a great
		// way to determine if the job is running on homogeneous
		// hardware. At some point, we should track this in the net
		// plugin and if we detect heterogeneity, start returning the
		// internal tuner defaults instead of our overrides. But for
		// now, we can take advantage of the fact that each AWS platform
		// has a different number of NICs per GPU and that a
		// heterogeneous job will have OFI_NCCL_FORCE_NUM_RAILS set by
		// the user as a key that this is a heterogeneous job. In that
		// case, abort out of the OFI tuner and use the internal tuner
		// (which does run after graph minimization, so will always
		// return the same answer on every process).
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
			      "Falling back to NCCL's tuner due to OFI_NCCL_FORCE_NUM_RAILS being set.");
		goto exit;
	}

	if (strcmp(platform_type, "p5.48xlarge") == 0 || strcmp(platform_type, "p5e.48xlarge") == 0) {
		tuner_platform = NCCL_OFI_TUNER_P5_P5E;
	} else if (strcmp(platform_type, "p5en.48xlarge") == 0) {
		tuner_platform = NCCL_OFI_TUNER_P5EN;
	} else if (strcmp(platform_type, "p6-b200.48xlarge") == 0) {
		tuner_platform = NCCL_OFI_TUNER_P6;
	} else {
		tuner_platform = NCCL_OFI_TUNER_UNKNOWN;
	}

	region_support = is_region_supported(tuner_platform, nRanks, nNodes);
	model_support = is_model_supported(tuner_platform, nRanks, nNodes);
	if (!region_support && !model_support) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
			      "NCCL_OFI_TUNER is not available for platform : %s, Fall back to NCCL's tuner",
			      platform_type);
		goto exit;
	}

	ctx = (nccl_ofi_tuner_context_t *)calloc(1, sizeof(nccl_ofi_tuner_context_t));
	if (ctx == NULL) {
		NCCL_OFI_WARN("Context allocation failed.");
		ret = ncclInternalError;
		goto exit;
	}

	/*
	 * We reach here. It means the folowing two conditions are met.
	 *  - "Internal" force is not set by env variable
	 *  - at least one of "Region" or "Model" tuner is supported for the given platform, nRanks and nNodes
	 */

	/*
	 * We choose "Region" over "Model" when both are supported.
	 * TUNER_TYPE env variable is ignored if the forced tuner type is not
	 * supported by the given platform, nRanks and nNodes.
	 */

	if (region_support && !(model_support && is_force_type_model)) {
		ctx->type = TUNER_TYPE::REGION;
		ctx->init_internal = region_init_internal;
		ctx->get_coll_info_internal_v3 = region_get_coll_info_internal_v3;
		ctx->get_coll_info_internal_v2 = region_get_coll_info_internal_v2;
		ctx->destroy_internal = region_destroy_internal;
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Region base Tuner is chosen for platform: %s", platform_type);
	} else {
		assert(model_support);
		ctx->type = TUNER_TYPE::MODEL;;
		ctx->init_internal = model_init_internal;
		ctx->get_coll_info_internal_v3 = model_get_coll_info_internal_v3;
		ctx->get_coll_info_internal_v2 = model_get_coll_info_internal_v2;
		ctx->destroy_internal = model_destroy_internal;
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Model base Tuner is chosen for platform: %s", platform_type);
	}

	ret = ctx->init_internal(ctx, tuner_platform, nRanks, nNodes);

	NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Tuner init: comm with %ld ranks and %ld nodes.", nRanks, nNodes);

exit:
	if (ret != ncclSuccess && ctx != NULL) {
		nccl_ofi_tuner_destroy((void *)ctx);
		ctx = NULL;
	}

	*context = (void *)ctx;
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ret;
}


static ncclResult_t nccl_ofi_tuner_init_v2(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
{
	/*
	 * NCCL parses these variables and applies user filters inside its
	 * current tuner logic. The tuner_v2 does not support setting these
	 * variables and so the internal tuner will be used instead.
	 */
	if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "The tuner plugin can not be loaded when "
				"explicitly choosing an algorithm or protocol "
				"with NCCL_ALGO/NCCL_PROTO. "
				"Defaulting to internal tuner.");
		*context = nullptr;
		return ncclSuccess;
	}
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, context);
}



static ncclResult_t nccl_ofi_tuner_get_coll_info(void *context,
						 ncclFunc_t collType,
						 size_t nBytes,
						 int numPipeOps,
						 float **collCostTable,
						 int numAlgo,
						 int numProto,
						 int *nChannels)
{
	ncclResult_t ret;

	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == NULL || ctx->get_coll_info_internal_v3 == NULL) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	ret = ctx->get_coll_info_internal_v3(ctx, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto, nChannels);

	return ret;
}

static const char* ofincclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

static const char* ofincclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

static ncclResult_t nccl_ofi_tuner_init_v6(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction,
                      ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_v6_t* constants)
{
        int minChunkSize [NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5] = {
                { ofi_nccl_min_chunk_tree_ll(), ofi_nccl_min_chunk_tree_ll128(), ofi_nccl_min_chunk_tree_simple()}, //Tree
                { ofi_nccl_min_chunk_ring_ll(), ofi_nccl_min_chunk_ring_ll128(), ofi_nccl_min_chunk_ring_simple()},  // Ring
                { 0, 0, 0}, { 0, 0, 0},  // Collnet Direct, Chain
                { ofi_nccl_min_chunk_nvls_ll(), ofi_nccl_min_chunk_nvls_ll128(), ofi_nccl_min_chunk_nvls_simple()}, // NVLS
                { ofi_nccl_min_chunk_nvls_tree_ll(), ofi_nccl_min_chunk_nvls_tree_ll128(), ofi_nccl_min_chunk_nvls_tree_simple()},  // NVLS Tree
                { 0, 0, ofi_nccl_min_chunk_pat_simple()}  // PAT
        };

        memcpy(constants->minChunkSize, minChunkSize, sizeof(constants->minChunkSize));

        ncclResult_t ret =  nccl_ofi_tuner_init(nRanks, nNodes, logFunction, context);

        for (int algo = 0; algo < NCCL_NUM_ALGORITHMS_V5; algo++) {
                for (int proto = 0; proto < NCCL_NUM_PROTOCOLS_V5; proto++) {
                        if (minChunkSize[algo][proto]) {
                                NCCL_OFI_WARN(" MIN CHUNKSIZE: ALGO %s PROTO %s CHUNKSIZE %d", ofincclAlgoToString(algo),
                                         ofincclProtoToString(proto), minChunkSize[algo][proto]);
                        }
                }
        }

        return ret;
}

static ncclResult_t nccl_ofi_tuner_get_coll_info_v6(void *context,
						 ncclFunc_t collType,
						 size_t nBytes,
						 int numPipeOps,
						 float **collCostTable,
						 int numAlgo,
						 int numProto,
                                                 int regBuff,
						 int *nChannels)
{
        return nccl_ofi_tuner_get_coll_info(context, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto, nChannels);
}

extern "C" const ncclTuner_v6_t ncclTunerPlugin_v6 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v6,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v6,
					   .finalize = nccl_ofi_tuner_destroy};

extern "C" const ncclTuner_v3_t ncclTunerPlugin_v3 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info,
					   .destroy = nccl_ofi_tuner_destroy};

/* **** V2 **** */
static ncclResult_t nccl_ofi_tuner_get_coll_info_v2(
	void *context, ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	ncclResult_t ret;

	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == NULL || ctx->get_coll_info_internal_v2 == NULL) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	ret = ctx->get_coll_info_internal_v2(ctx,
					     collType,
					     nBytes,
					     collNetSupport,
					     nvlsSupport,
					     numPipeOps,
					     algorithm,
					     protocol,
					     nChannels);

	return ret;
}

extern "C" const ncclTuner_v2_t ncclTunerPlugin_v2 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v2,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v2,
					   .destroy = nccl_ofi_tuner_destroy};

/* **** V1 ****
 * The tuner v1 API is missing a mechanism to pass around context after
 * initialization. For now, init a plugin-global context once.
 */
static nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx_internal;

static ncclResult_t nccl_ofi_tuner_destroy_v1(void)
{
	void *context = NULL;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (nccl_ofi_tuner_ctx_internal != NULL) {
		/* Prevent other threads from freeing a dangling global ctx */
		context = (void *)nccl_ofi_tuner_ctx_internal;
		nccl_ofi_tuner_ctx_internal = NULL;
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return nccl_ofi_tuner_destroy(context);
}

static ncclResult_t nccl_ofi_tuner_init_v1(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction)
{
	if (nccl_ofi_tuner_ctx_internal != NULL) {
		/* Repeated init call, the tuner is already initialized.
		 * Destroy it, as it may have been initialized with different
		 * parameters.
		 */
		if (nccl_ofi_tuner_destroy_v1() != ncclSuccess) {
			NCCL_OFI_WARN(
				"Failed to destroy an existing tuner context.");
		}
	}

	/*
	 * NCCL parses these variables and applies user filters inside its
	 * current tuner logic. The tuner_v1 does not support setting these
	 * variables and so the internal tuner will be used instead.
	 */
	if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "The tuner plugin can not be loaded when "
				"explicitly choosing an algorithm or protocol "
				"with NCCL_ALGO/NCCL_PROTO. "
				"Defaulting to internal tuner.");
		nccl_ofi_tuner_destroy_v1();
		return ncclSuccess;
	}
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, (void **)&nccl_ofi_tuner_ctx_internal);
}

static ncclResult_t nccl_ofi_tuner_get_coll_info_v1(
	ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	return nccl_ofi_tuner_get_coll_info_v2(nccl_ofi_tuner_ctx_internal,
					    collType,
					    nBytes,
					    collNetSupport,
					    nvlsSupport,
					    numPipeOps,
					    algorithm,
					    protocol,
					    nChannels);
}

extern "C" const ncclTuner_v1_t ncclTunerPlugin_v1 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v1,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v1,
					   .destroy = nccl_ofi_tuner_destroy_v1};
