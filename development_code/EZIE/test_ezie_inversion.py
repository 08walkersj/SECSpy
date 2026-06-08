from SECSpy.EZIE_inversion import inversion
import numpy as np
inv= inversion('../../Data/ezie_l2_20250518_234515_sva_v001_r001.nc4',
               np.datetime64('2025-05-19T00:00:00'))
inv.get_sat_vector()
inv.create_grid()
inv.create_SECS()
inv.download_data('08walkersj', save=True)
inv.plot_pass()