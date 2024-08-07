from tqdm import tqdm
from find_markers import *

def load_group_file(groups_file, betas):
    validate_single_file(groups_file)
    validate_file_list(betas)
    gf = load_gfile_helper(groups_file)
    gf['full_path'] = match_prefix_to_bin(gf['fname'], betas, '.uxm')
    return gf


def get_uxm_table(blocks_df, gf_nodup, min_cov, keepinds, only_hypo):

    if blocks_df.empty:
        return pd.DataFrame()

    z = np.zeros((blocks_df.shape[0], gf_nodup.shape[0]), dtype=float)
    dfU = pd.DataFrame(data=z, columns=gf_nodup.fname.values)

    start, end = keepinds
    if (end - start) > blocks_df.shape[0]:
        end = start + blocks_df.shape[0]
    if gf_nodup.shape[0] > 50:
        it = tqdm(gf_nodup.iterrows(), total=gf_nodup.shape[0])
    else:
        it = gf_nodup.iterrows()

    um_col = 0 if only_hypo else 2
    for ind, row in it:
        with open(row['full_path'], 'rb') as f:
            f.seek(start * 3)
            data = np.fromfile(f, dtype=np.uint8, count=((end - start) * 3)).reshape((-1, 3)).astype(float)
        dfU[row['fname']] = table2vec(data, min_cov, um_col)

    return pd.concat([blocks_df, dfU], axis=1)


def table2vec(table, cov_thresh, um_col=0):
    low_cov = (table.sum(axis=1) < cov_thresh)
    table[low_cov, :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = table[:, um_col] / table.sum(axis=1)
    return x


class UXMMarkerFinder:
    def __init__(self, args):
        self.args = args
        self.df = pd.DataFrame()
        self.verbose = args.verbose
        self.tg_names = []
        self.bg_names = []
        self.inds_dict = {}
        self.chunk_count = 0
        self.nr_chunks = 1

        # validate output dir:
        mkdirp(args.out_dir)

        # load groups
        self.gf = load_group_file(args.groups_file, args.betas)
        groups = sorted(self.gf['group'].unique())
        self.targets = get_validate_targets(self.args.targets, groups)
        self.background = get_validate_targets(self.args.background, groups)
        self.res = {t: pd.DataFrame() for t in self.targets}
        self.inds_dict, self.gf = set_bg_tg_names(self.gf, self.targets, self.background)
        self.gf_nodup = self.gf.drop_duplicates(subset='fname').reset_index(drop=True)

    def run(self):

        # first dump the param file
        self.dump_params()

        # load all blocks
        blocks = self.load_blocks()
        if blocks.empty:
            eprint('Empty block set. Abort')
            return

        # load data in chunks
        chunk_size = self.args.chunk_size
        self.nr_chunks = ceil(blocks.shape[0] / chunk_size)
        if self.verbose:
            eprint(f'processing data in {self.nr_chunks} chunks...')
        for start in range(0, blocks.shape[0], chunk_size):
            self.keepinds = [start, start + chunk_size]
            self.proc_chunk(blocks.iloc[start:start + chunk_size])

        # dump results:
        for target in self.targets:
            self.group = target
            self.tg_names, self.bg_names = self.inds_dict[self.group]
            self.dump_results(self.res[target].reset_index(drop=True))

    def proc_chunk(self, blocks_df):
        self.df = self.load_data_chunk(blocks_df)

        for group in self.targets:
            # eprint(f'target: {group}')
            self.group = group
            tf = self.find_group_markers()
            self.res[group] = pd.concat([self.res[group], tf])

    #############################
    #                           #
    #       Load methods        #
    #                           #
    #############################

    def load_blocks(self):
        # load blocks file and filter it by CpG and bg length

        df = load_blocks_file(self.args.blocks_path, anno=True)
        if df.empty:
            return df
        orig_nr_blocks = df.shape[0]

        # filter by lenCpG
        df['lenCpG'] = df['endCpG'] - df['startCpG']
        df = df[df['lenCpG'] >= self.args.min_cpg]
        df = df[df['lenCpG'] <= self.args.max_cpg]

        # filter by len in bp
        df['len'] = df['end'] - df['start']
        df = df[df['len'] >= self.args.min_bp]
        df = df[df['len'] <= self.args.max_bp]

        df.reset_index(drop=True, inplace=True)

        # print stats
        if self.verbose:
            eprint(f'loaded {orig_nr_blocks:,} blocks')
            if df.shape[0] != orig_nr_blocks:
                eprint(f'droppd to {df.shape[0]:,} ')

        return df

    def load_data_chunk(self, blocks_df):
        # load methylation data from beta files collapsed to the blocks in blocks_df
        if self.verbose:
            self.chunk_count += 1
            nr_samples = len(self.gf['fname'].unique())
            eprint(f'{self.chunk_count}/{self.nr_chunks} ) ' \
                   f'loading data for {blocks_df.shape[0]:,} blocks over' \
                   f' {nr_samples} samples...')

        return get_uxm_table(blocks_df=blocks_df.copy().reset_index(drop=True),
                             gf_nodup=self.gf_nodup,
                             min_cov=self.args.min_cov,
                             keepinds=self.keepinds,
                             only_hypo=self.args.only_hypo)

    #############################
    #                           #
    #       Finding markers     #
    #                           #
    #############################

    def find_group_markers(self):
        if self.df.empty:
            return self.df

        # load context
        self.tg_names, self.bg_names = self.inds_dict[self.group]
        tf = self.df.copy()

        # filter blocks by coverage:
        keep_tg = (tf[self.tg_names].notna().sum(axis=1) / len(self.tg_names)) >= (1 - self.args.na_rate_tg)
        keep_bg = (tf[self.bg_names].notna().sum(axis=1) / len(self.bg_names)) >= (1 - self.args.na_rate_bg)
        tf = tf.loc[keep_tg & keep_bg, :].reset_index(drop=True)

        # find markers:
        tf = self.find_U_markers(tf)

        # T-test
        tf = self.ttest(tf)
        return tf

    def ttest(self, tf):
        try:
            tf['ttest'] = np.nan
            # if n=1 for both bg and tg samples, break
            if len(self.tg_names) == len(self.bg_names) == 1:
                return tf
            from scipy.stats import ttest_ind, ttest_1samp
            # test for n=1
            if len(self.tg_names) == 1:
                r = ttest_1samp(tf[self.bg_names], tf[self.tg_names].values, axis=1, nan_policy='omit')
            elif len(self.bg_names) == 1:
                r = ttest_1samp(tf[self.tg_names], tf[self.bg_names].values, axis=1, nan_policy='omit')
            # test for n>1
            else:
                r = ttest_ind(tf[self.tg_names], tf[self.bg_names], axis=1, nan_policy='omit')
            tf['ttest'] = r.pvalue
            tf = tf[tf['ttest'] <= self.args.pval].reset_index(drop=True)
        except ModuleNotFoundError as e:
            eprint(f'[wt fm] WARNING: scipy is not installed. T-test is not performed.')
        except Exception as e:
            eprint(f'[wt fm] WARNING: Eception occured while computing T-test. T-test is not performed.')
        return tf

    def find_X_markers(self, tf):

        tfX = tf.copy()

        # Compute maxmin
        tfX['delta_maxmin'] = tfX[self.tg_names].min(axis=1) - tfX[self.bg_names].max(axis=1)

        # Compute means for target and background
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tfX['tg_mean'] = np.nanmean(tfX[self.tg_names], axis=1)
            tfX['bg_mean'] = np.nanmean(tfX[self.bg_names], axis=1)
        tfX['delta_means'] = tfX['tg_mean'] - tfX['bg_mean']

        # filter by mean thresholds
        keep_umt = (tfX['tg_mean'] >= self.args.unmeth_mean_thresh)
        keep_mmt = (tfX['bg_mean'] <= self.args.meth_mean_thresh)
        keep_delta_mean = (tfX['delta_means'] >= self.args.delta_means)
        tfX = tfX.loc[keep_umt & keep_mmt & keep_delta_mean, :].reset_index(drop=True)

        if tfX.empty:
            return tfX

        # Compute quantiles for target and background
        tfX['tg_quant'] = np.nanquantile(tfX[self.tg_names], 1 - self.args.tg_quant, axis=1)
        tfX['bg_quant'] = np.nanquantile(tfX[self.bg_names], self.args.bg_quant, axis=1)
        tfX['delta_quants'] = tfX['tg_quant'] - tfX['bg_quant']

        # filter by quantile thresholds
        keep_uqt = (tfX['tg_quant'] >= self.args.unmeth_quant_thresh)
        keep_mqt = (tfX['bg_quant'] <= self.args.meth_quant_thresh)
        keep_delta_quants = (tfX['delta_quants'] >= self.args.delta_quants)
        tfX = tfX.loc[keep_uqt & keep_mqt & keep_delta_quants, :].reset_index(drop=True)

        if tfX.empty:
            return tfX

        return tfX

    def find_U_markers(self, tf):
        # look for 'U' markers
        tfU = self.find_X_markers(tf)
        if tfU.empty:
            return tfU

        tfU['direction'] = 'U'
        if self.args.only_hyper:
            tfU['direction'] = 'M'
        return tfU


    #############################
    #                           #
    #       Dump methods        #
    #                           #
    #############################

    def dump_results(self, tf):
        eprint(f'Number of markers found: {tf.shape[0]:,}')
        if self.args.sort_by:
            tf = tf.sort_values(by=self.args.sort_by, ascending=False)
        if self.args.top:
            tf = tf.head(self.args.top)
        tf['target'] = self.group
        tf['lenCpG'] = tf['lenCpG'].astype(str) + 'CpGs'
        tf['bp'] = (tf['end'] - tf['start']).astype(str) + 'bp'
        tf['region'] = bed2reg(tf)
        cols_to_dump = ['chr', 'start', 'end', 'startCpG', 'endCpG',
                        'target', 'region', 'lenCpG', 'bp', 'tg_mean',
                        'bg_mean', 'delta_means', 'delta_quants', 'delta_maxmin',
                        'ttest', 'direction']
        if 'anno' in list(tf.columns) and 'gene' in list(tf.columns):
            cols_to_dump += ['anno', 'gene']
        if not tf.empty:
            tf = tf[cols_to_dump]
        else:
            tf = pd.DataFrame(columns=cols_to_dump)
        tf = tf.rename(columns={"chr": "#chr"})
        self.dump_single_df(tf)

    def dump_single_df(self, df):
        """ Dump the results, with comments header """

        outpath = op.join(self.args.out_dir, f'Markers.{self.group}.bed')
        eprint(f'dumping to {outpath}')
        df_mode = 'w'

        # write header:
        if self.args.header:
            # write header (comments):
            with open(outpath, 'w') as f:
                for sample in sorted(self.tg_names):
                    f.write(f'#> {sample}\n' )
                for sample in sorted(self.bg_names):
                    f.write(f'#< {sample}\n' )
            df_mode = 'a'

        # dump df:
        df.to_csv(outpath, index=None, sep='\t', mode=df_mode, header=True, na_rep='NA', float_format='%.3g')

    def dump_params(self):
        """ Dump a parameter file """
        outpath = op.join(self.args.out_dir, 'params.txt')
        with open(outpath, 'w') as f:
            for key in vars(self.args):
                val = getattr(self.args, key)
                if key == 'beta_list_file':
                    val = None
                if key == 'betas':
                    val = ' '.join(val)
                if key == 'targets' and val is not None:
                    val = ' '.join(val)
                f.write(f'{key}:{val}\n' )
                # f.write(f'#> {sample}\n' )
        eprint(f'dumped parameter file to {outpath}')


def main():
    """
    Find differentially methylated blocks based on U reads
    """
    params = MFParams(parse_args())
    UXMMarkerFinder(params).run()


if __name__ == '__main__':
    main()
