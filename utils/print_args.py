def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Dataset:":<20}{args.dataset:<20}{"Window Sliding:":<20}{args.window_sliding:<20}')
    print(f'  {"Data Division:":<20}{(args.data_division if args.data_division is not None else "N/A"):<20}')
    print()

    print("\033[1m" + "Data Augmentation" + "\033[0m")
    print(f'  {"Soft Replacing:":<20}{args.soft_replacing:<20}{"Flip Replacing Interval:":<20}{args.flip_replacing_interval:<20}')
    print(f'  {"Uniform Replacing:":<20}{args.uniform_replacing:<20}{"Peak Noising:":<20}{args.peak_noising:<20}')
    print(f'  {"Length Adjusting:":<20}{args.length_adjusting:<20}{"White Noising:":<20}{args.white_noising:<20}')
    print()

    print("\033[1m" + "Experimental Log" + "\033[0m")
    print(f'  {"Summary Steps:":<20}{args.summary_steps:<20}{"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
    print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
    print()

    print("\033[1m" + "Masking Ratio" + "\033[0m")
    print(f'  {"Masking Ratio:":<20}{args.mask_ratio:<20}')
    print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"N Features:":<20}{args.input_encoder_len:<20}{"Patch Size:":<20}{args.patch_size:<20}')
    print(f'  {"E Layers:":<20}{args.e_layers:<20}{"D Model:":<20}{args.d_model:<20}')
    print(f'  {"Dropout:":<20}{args.dropout:<20}{"Replacing Rate Max:":<20}{args.replacing_rate_max:<20}')
    print(f'  {"Replacing Weight:":<20}{args.replacing_weight:<20}{"N Heads:":<20}{args.n_heads:<20}')
    print()

    print("\033[1m" + "Optimization" + "\033[0m")
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Loss:":<20}{args.loss:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Grad Clip Norm:":<20}{args.grad_clip_norm:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"GPU ID:":<20}{args.gpu_id:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}')
    print()