from .config import (get_chunk_size, get_max_threads, get_num_threads,
                     get_threading_layer, set_chunk_size, set_default_threads,
                     set_num_threads, set_threading_layer)

set_threading_layer(thread_layer='tbb')
