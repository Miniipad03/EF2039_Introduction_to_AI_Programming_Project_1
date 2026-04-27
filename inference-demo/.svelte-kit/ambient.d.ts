
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/private';
 * 
 * console.log(ENVIRONMENT); // => "production"
 * console.log(PUBLIC_BASE_URL); // => throws error during build
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/private' {
	export const DALI_BUILD: string;
	export const PYTHON_BASIC_REPL: string;
	export const LIBRARY_PATH: string;
	export const BACKENDAI_SERVICE_PORTS: string;
	export const PYTORCH_BUILD_NUMBER: string;
	export const PYTHONIOENCODING: string;
	export const C_INCLUDE_PATH: string;
	export const PIP_CONSTRAINT: string;
	export const PIP_DEFAULT_TIMEOUT: string;
	export const PYTORCH_HOME: string;
	export const npm_config_user_agent: string;
	export const BACKENDAI_CLUSTER_SIZE: string;
	export const TF_CONFIG: string;
	export const EFA_VERSION: string;
	export const MPLBACKEND: string;
	export const BACKENDAI_ACCESS_KEY: string;
	export const CUSOLVER_VERSION: string;
	export const HOSTNAME: string;
	export const SSH_AGENT_PID: string;
	export const CUDA_ARCH_LIST: string;
	export const COCOAPI_VERSION: string;
	export const GIT_ASKPASS: string;
	export const npm_node_execpath: string;
	export const BACKENDAI_KERNEL_ID: string;
	export const OPENBLAS_NUM_THREADS: string;
	export const SHLVL: string;
	export const LD_LIBRARY_PATH: string;
	export const BROWSER: string;
	export const npm_config_noproxy: string;
	export const LOCAL_USER_ID: string;
	export const BACKENDAI_CLUSTER_HOST: string;
	export const BACKENDAI_USER_UUID: string;
	export const LOCAL_GROUP_ID: string;
	export const HOME: string;
	export const ALPHA_NUMERIC_VAL: string;
	export const OLDPWD: string;
	export const TERM_PROGRAM_VERSION: string;
	export const AWS_OFI_NCCL_VERSION: string;
	export const VSCODE_IPC_HOOK_CLI: string;
	export const npm_package_json: string;
	export const PYTHONUNBUFFERED: string;
	export const TORCH_NCCL_USE_COMM_NONBLOCKING: string;
	export const BACKENDAI_PREOPEN_PORTS: string;
	export const PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: string;
	export const VSCODE_GIT_ASKPASS_MAIN: string;
	export const DALI_URL_SUFFIX: string;
	export const MODEL_OPT_VERSION: string;
	export const VSCODE_GIT_ASKPASS_NODE: string;
	export const BACKENDAI_CLUSTER_REPLICAS: string;
	export const npm_config_userconfig: string;
	export const npm_config_local_prefix: string;
	export const PYDEVD_DISABLE_FILE_VALIDATION: string;
	export const ENV: string;
	export const BUNDLED_DEBUGPY_PATH: string;
	export const RDMACORE_VERSION: string;
	export const NVJPEG_VERSION: string;
	export const NVIDIA_BUILD_ID: string;
	export const BACKENDAI_CLUSTER_HOSTS: string;
	export const NCCL_CUDA_PATH: string;
	export const COLORTERM: string;
	export const CUFILE_VERSION: string;
	export const TF_MIN_GPU_MULTIPROCESSOR_COUNT: string;
	export const COLOR: string;
	export const CUDA_VERSION: string;
	export const CPLUS_INCLUDE_PATH: string;
	export const TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: string;
	export const CUBLAS_VERSION: string;
	export const TORCH_CUDA_ARCH_LIST: string;
	export const NSIGHT_SYSTEMS_VERSION: string;
	export const CAL_VERSION: string;
	export const CUBLASMP_VERSION: string;
	export const OPAL_PREFIX: string;
	export const CUDA_MODULE_LOADING: string;
	export const NVJITLINK_VERSION: string;
	export const NVIDIA_REQUIRE_CUDA: string;
	export const TRT_VERSION: string;
	export const BACKENDAI_USER_EMAIL: string;
	export const GDRCOPY_VERSION: string;
	export const PYTORCH_BUILD_VERSION: string;
	export const _: string;
	export const npm_config_prefix: string;
	export const npm_config_npm_version: string;
	export const BACKENDAI_CLUSTER_LOCAL_RANK: string;
	export const NVSHMEM_VERSION: string;
	export const NVIDIA_DRIVER_CAPABILITIES: string;
	export const POLYGRAPHY_VERSION: string;
	export const PIP_BREAK_SYSTEM_PACKAGES: string;
	export const CURAND_VERSION: string;
	export const MOFED_VERSION: string;
	export const BACKENDAI_SESSION_NAME: string;
	export const TERM: string;
	export const npm_config_cache: string;
	export const PIP_IGNORE_INSTALLED: string;
	export const TRANSFORMER_ENGINE_VERSION: string;
	export const NVIDIA_PYTORCH_VERSION: string;
	export const npm_config_node_gyp: string;
	export const PATH: string;
	export const PYTORCH_VERSION: string;
	export const NODE: string;
	export const npm_package_name: string;
	export const JUPYTER_PORT: string;
	export const CUDA_DRIVER_VERSION: string;
	export const _CUDA_COMPAT_STATUS: string;
	export const NVFUSER_BUILD_VERSION: string;
	export const DISPLAY: string;
	export const BACKENDAI_USER_NAME: string;
	export const VSCODE_DEBUGPY_ADAPTER_ENDPOINTS: string;
	export const VSCODE_PROXY_URI: string;
	export const NVIDIA_PRODUCT_NAME: string;
	export const BACKENDAI_KERNEL_IMAGE: string;
	export const LD_PRELOAD: string;
	export const LANG: string;
	export const PYTHONSTARTUP: string;
	export const NPP_VERSION: string;
	export const TENSORBOARD_PORT: string;
	export const NVPL_LAPACK_MATH_MODE: string;
	export const VSCODE_GIT_IPC_HANDLE: string;
	export const NODE_EXEC_PATH: string;
	export const TERM_PROGRAM: string;
	export const npm_lifecycle_script: string;
	export const CUFFT_VERSION: string;
	export const SSH_AUTH_SOCK: string;
	export const CUDNN_VERSION: string;
	export const NSIGHT_COMPUTE_VERSION: string;
	export const DALI_VERSION: string;
	export const BACKENDAI_SESSION_ID: string;
	export const DEBIAN_FRONTEND: string;
	export const SHELL: string;
	export const CUDNN_FRONTEND_VERSION: string;
	export const OPENMPI_VERSION: string;
	export const TRTOSS_VERSION: string;
	export const npm_package_version: string;
	export const npm_lifecycle_event: string;
	export const NVFUSER_VERSION: string;
	export const OMPI_MCA_coll_hcoll_enable: string;
	export const CUSPARSE_VERSION: string;
	export const UCC_CL_BASIC_TLS: string;
	export const NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS: string;
	export const CODESERVER: string;
	export const VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
	export const BASH_ENV: string;
	export const npm_config_globalconfig: string;
	export const npm_config_init_module: string;
	export const BACKENDAI_CLUSTER_ROLE: string;
	export const PWD: string;
	export const LC_ALL: string;
	export const CUDA_HOME: string;
	export const npm_execpath: string;
	export const BACKENDAI_CLUSTER_IDX: string;
	export const CUSPARSELT_VERSION: string;
	export const npm_config_global_prefix: string;
	export const PYTHONPATH: string;
	export const OMP_NUM_THREADS: string;
	export const _CUDA_COMPAT_PATH: string;
	export const NPROC: string;
	export const npm_command: string;
	export const NVIDIA_VISIBLE_DEVICES: string;
	export const NCCL_VERSION: string;
	export const OPENUCX_VERSION: string;
	export const HPCX_VERSION: string;
	export const INIT_CWD: string;
	export const EDITOR: string;
	export const NODE_ENV: string;
}

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/public';
 * 
 * console.log(ENVIRONMENT); // => throws error during build
 * console.log(PUBLIC_BASE_URL); // => "http://site.com"
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * 
 * console.log(env.ENVIRONMENT); // => "production"
 * console.log(env.PUBLIC_BASE_URL); // => undefined
 * ```
 */
declare module '$env/dynamic/private' {
	export const env: {
		DALI_BUILD: string;
		PYTHON_BASIC_REPL: string;
		LIBRARY_PATH: string;
		BACKENDAI_SERVICE_PORTS: string;
		PYTORCH_BUILD_NUMBER: string;
		PYTHONIOENCODING: string;
		C_INCLUDE_PATH: string;
		PIP_CONSTRAINT: string;
		PIP_DEFAULT_TIMEOUT: string;
		PYTORCH_HOME: string;
		npm_config_user_agent: string;
		BACKENDAI_CLUSTER_SIZE: string;
		TF_CONFIG: string;
		EFA_VERSION: string;
		MPLBACKEND: string;
		BACKENDAI_ACCESS_KEY: string;
		CUSOLVER_VERSION: string;
		HOSTNAME: string;
		SSH_AGENT_PID: string;
		CUDA_ARCH_LIST: string;
		COCOAPI_VERSION: string;
		GIT_ASKPASS: string;
		npm_node_execpath: string;
		BACKENDAI_KERNEL_ID: string;
		OPENBLAS_NUM_THREADS: string;
		SHLVL: string;
		LD_LIBRARY_PATH: string;
		BROWSER: string;
		npm_config_noproxy: string;
		LOCAL_USER_ID: string;
		BACKENDAI_CLUSTER_HOST: string;
		BACKENDAI_USER_UUID: string;
		LOCAL_GROUP_ID: string;
		HOME: string;
		ALPHA_NUMERIC_VAL: string;
		OLDPWD: string;
		TERM_PROGRAM_VERSION: string;
		AWS_OFI_NCCL_VERSION: string;
		VSCODE_IPC_HOOK_CLI: string;
		npm_package_json: string;
		PYTHONUNBUFFERED: string;
		TORCH_NCCL_USE_COMM_NONBLOCKING: string;
		BACKENDAI_PREOPEN_PORTS: string;
		PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: string;
		VSCODE_GIT_ASKPASS_MAIN: string;
		DALI_URL_SUFFIX: string;
		MODEL_OPT_VERSION: string;
		VSCODE_GIT_ASKPASS_NODE: string;
		BACKENDAI_CLUSTER_REPLICAS: string;
		npm_config_userconfig: string;
		npm_config_local_prefix: string;
		PYDEVD_DISABLE_FILE_VALIDATION: string;
		ENV: string;
		BUNDLED_DEBUGPY_PATH: string;
		RDMACORE_VERSION: string;
		NVJPEG_VERSION: string;
		NVIDIA_BUILD_ID: string;
		BACKENDAI_CLUSTER_HOSTS: string;
		NCCL_CUDA_PATH: string;
		COLORTERM: string;
		CUFILE_VERSION: string;
		TF_MIN_GPU_MULTIPROCESSOR_COUNT: string;
		COLOR: string;
		CUDA_VERSION: string;
		CPLUS_INCLUDE_PATH: string;
		TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: string;
		CUBLAS_VERSION: string;
		TORCH_CUDA_ARCH_LIST: string;
		NSIGHT_SYSTEMS_VERSION: string;
		CAL_VERSION: string;
		CUBLASMP_VERSION: string;
		OPAL_PREFIX: string;
		CUDA_MODULE_LOADING: string;
		NVJITLINK_VERSION: string;
		NVIDIA_REQUIRE_CUDA: string;
		TRT_VERSION: string;
		BACKENDAI_USER_EMAIL: string;
		GDRCOPY_VERSION: string;
		PYTORCH_BUILD_VERSION: string;
		_: string;
		npm_config_prefix: string;
		npm_config_npm_version: string;
		BACKENDAI_CLUSTER_LOCAL_RANK: string;
		NVSHMEM_VERSION: string;
		NVIDIA_DRIVER_CAPABILITIES: string;
		POLYGRAPHY_VERSION: string;
		PIP_BREAK_SYSTEM_PACKAGES: string;
		CURAND_VERSION: string;
		MOFED_VERSION: string;
		BACKENDAI_SESSION_NAME: string;
		TERM: string;
		npm_config_cache: string;
		PIP_IGNORE_INSTALLED: string;
		TRANSFORMER_ENGINE_VERSION: string;
		NVIDIA_PYTORCH_VERSION: string;
		npm_config_node_gyp: string;
		PATH: string;
		PYTORCH_VERSION: string;
		NODE: string;
		npm_package_name: string;
		JUPYTER_PORT: string;
		CUDA_DRIVER_VERSION: string;
		_CUDA_COMPAT_STATUS: string;
		NVFUSER_BUILD_VERSION: string;
		DISPLAY: string;
		BACKENDAI_USER_NAME: string;
		VSCODE_DEBUGPY_ADAPTER_ENDPOINTS: string;
		VSCODE_PROXY_URI: string;
		NVIDIA_PRODUCT_NAME: string;
		BACKENDAI_KERNEL_IMAGE: string;
		LD_PRELOAD: string;
		LANG: string;
		PYTHONSTARTUP: string;
		NPP_VERSION: string;
		TENSORBOARD_PORT: string;
		NVPL_LAPACK_MATH_MODE: string;
		VSCODE_GIT_IPC_HANDLE: string;
		NODE_EXEC_PATH: string;
		TERM_PROGRAM: string;
		npm_lifecycle_script: string;
		CUFFT_VERSION: string;
		SSH_AUTH_SOCK: string;
		CUDNN_VERSION: string;
		NSIGHT_COMPUTE_VERSION: string;
		DALI_VERSION: string;
		BACKENDAI_SESSION_ID: string;
		DEBIAN_FRONTEND: string;
		SHELL: string;
		CUDNN_FRONTEND_VERSION: string;
		OPENMPI_VERSION: string;
		TRTOSS_VERSION: string;
		npm_package_version: string;
		npm_lifecycle_event: string;
		NVFUSER_VERSION: string;
		OMPI_MCA_coll_hcoll_enable: string;
		CUSPARSE_VERSION: string;
		UCC_CL_BASIC_TLS: string;
		NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS: string;
		CODESERVER: string;
		VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
		BASH_ENV: string;
		npm_config_globalconfig: string;
		npm_config_init_module: string;
		BACKENDAI_CLUSTER_ROLE: string;
		PWD: string;
		LC_ALL: string;
		CUDA_HOME: string;
		npm_execpath: string;
		BACKENDAI_CLUSTER_IDX: string;
		CUSPARSELT_VERSION: string;
		npm_config_global_prefix: string;
		PYTHONPATH: string;
		OMP_NUM_THREADS: string;
		_CUDA_COMPAT_PATH: string;
		NPROC: string;
		npm_command: string;
		NVIDIA_VISIBLE_DEVICES: string;
		NCCL_VERSION: string;
		OPENUCX_VERSION: string;
		HPCX_VERSION: string;
		INIT_CWD: string;
		EDITOR: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://example.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.ENVIRONMENT); // => undefined, not public
 * console.log(env.PUBLIC_BASE_URL); // => "http://example.com"
 * ```
 * 
 * ```
 * 
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
