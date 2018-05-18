import time

import numpy as np
from qcodes import (
    MultiParameter, InstrumentChannel, VisaInstrument, validators as vals,
    DataArray
)

# get_parser shorthands
def qstr(s):
    '''convert quoted string to string. just strips the quotes currently.'''
    return s[1:-1]
def ibool(v):
    '''convert str -> int -> bool'''
    return bool(int(v))

class DPOCurve(MultiParameter):
    def __init__(self, name, source, command='CURVE', **kwargs):
        '''
        Input
        -----
        parent: `Instrument`
            Object used to communicate with the instrument
        name: `str`
            Name of the Parameter
        source: `str`
            Channel, math or reference waveform to retrieve.
            One of CH<x>, MATH<x>, REF<x>, DIGITALALL.

        To do
        -----
        * Point format can be Y or ENVelope. ENVelope should be returned as a
          MultiParameter with min and max arrays instead.
        '''
        super().__init__(name, names=(name,), shapes=((),), **kwargs)
        self.source = source
        self.command = command

    def preamble(self):
        '''
        Set source, request and format waveform output preamble.
        '''
        preamble_keys_parsers = list(zip(*[
            ('BYT_NR', int), ('BIT_NR', int), ('ENCDG', str), ('BN_FMT', str),
            ('BYT_OR', str), ('WFID', qstr), ('NR_PT', int), ('PT_FMT', str),
            ('XUNIT', qstr), ('XINCR', float), ('XZERO', float), ('PT_OFF', int), 
            ('YUNIT', qstr), ('YMULT', float), ('YOFF', float), ('YZERO', float), 
            ('NR_FR', int), ('PT_OR', str), ('ACQLEN', int), ('FASTFRAME', ibool), 
            ('SUMFRAME', str)
        ]))
        preamble_vals = self._instrument.ask(
            ':DATA:SOURCE {}; '.format(self.source) + 
            ':WFMO?; :WFMO:PT_OR?; '
            ':HOR:ACQLENGTH?; :HOR:FAST:STATE?; SUMFRAME?'
        )
        preamble = dict((key, parser(value)) for value, key, parser 
                        in zip(preamble_vals.split(';'), *preamble_keys_parsers))
        preamble['PIXMAP'] = (preamble['PT_OR'] == 'COLUMN')
        return preamble

    def update(self, force=False, preamble=None):
        '''
        Update shape, label, unit and setpoints if required.

        Input
        -----
        force: `bool`
            If True, always update.
        preamble: `dict`, optional
            If given, use as the waveform preamble instead of retrieving 
            the preamble from the instrument.
        '''
        # check for changes in preamble
        if preamble is None:
            preamble = self.preamble()
        if not force:
            old_preamble = getattr(self, '_cached_preamble', {})
            if preamble == old_preamble:
                return
        self._cached_preamble = preamble
        self._update(preamble)

    def _update(self, pre):
        '''Update shape, label unit and setpoints from the waveform preamble.'''
        # determine label and data unit
        # if the point format is ENVelope, the resolution is halved and the 
        # raw data alternate between min and max for each sampling interval
        if pre['PT_FMT'] == 'ENV':
            self.names = ('min', 'max')
            self.labels = ('{}, minimum'.format(pre['WFID']),
                           '{}, maximum'.format(pre['WFID']))
            noutputs = 2
        else:
            self.names = (self.name,)
            self.labels = (pre['WFID'],)
            noutputs = 1
        self.units = (pre['YUNIT'],)*noutputs
        # determine setpoints
        if pre['PIXMAP']:
            npoints = 1000
            ptoffset = pre['PT_OFF'] * pre['ACQLEN'] // npoints
        else:
            npoints = pre['NR_PT'] // noutputs
            ptoffset = pre['PT_OFF'] // noutputs
        sp_times = DataArray(
            name='X', unit=pre['XUNIT'], is_setpoint=True, 
            preset_data=(pre['XINCR']*noutputs*np.arange(-ptoffset, npoints-ptoffset) +
                         pre['XZERO'])
        )
        if not pre['FASTFRAME']:
            # output is 1d if FastFrame is disabled
            self.shapes = ((npoints,),)*noutputs
            self.setpoints = ((sp_times,),)*noutputs
        else:
            # output is 2d if FastFrame is enabled
            nframes = pre['NR_FR']
            self.shapes = ((nframes, npoints),)*noutputs
            sp_frames = DataArray(
                name='frame', unit='', is_setpoint=True, 
                preset_data=np.arange(nframes)
            )
            sp_times.nest(len(sp_frames))
            self.setpoints = ((sp_frames, sp_times),)*noutputs

    def prepare(self, data_start=1, data_stop=(1<<31)-1, frame_start=1, 
                frame_stop=(1<<31)-1):
        '''
        Prepare curve for data acquisition.

        Sets data start/stop and frame start/stop on the instrument.
        Updates shape, label, units and setpoints via update().

        Input
        -----
        data_start, data_stop: `int`
            First and last sample to transfer
        frame_start, frame_stop: `int`
            First and last frame to transfer in FastFrame mode
        '''
        # set data source and range
        self._instrument.write('; '.join((
            'DATA:SOURCE {}'.format(self.source),
            'START {}'.format(data_start),
            'STOP {}'.format(data_stop),
            'FRAMESTART {}'.format(frame_start),
            'FRAMESTOP {}'.format(frame_stop)
        )))
        self.update()

    def get_raw(self, raw=False):
        '''
        Retrieve data from instrument.

        * acq_mode==sample, hires, average: ok
        * acq_mode==peakdetect, envelope: 0::2 and 1::2 are min/max
        * acq_mode==wfmdb: requires scale=False
        '''
        # check and set data encoding
        # * ASCII in fast and wfmdb modes (data corruption happens otherwise)
        # * FASTEST otherwise, which is int or float depending on source
        # * default width is 8bit, increase to 16bit when
        #   * acquisition mode average is used
        #   * fast frame with an averaged summary frame is used
        pre = self.preamble()
        if pre['PIXMAP']:
            if pre['ENCDG'] != 'ASC':
                self._instrument.write('DATA:ENC ASCII')
                pre = self.preamble()
        else:
            byt_nr = 2 if (('Average mode' in pre['WFID']) or \
                           pre['FASTFRAME'] and (pre['SUMFRAME'] == 'AVE')) else \
                     1
            if (
                (pre['ENCDG'] != 'BIN') or
                (pre['BN_FMT'] != 'FP') and (pre['BYT_NR'] != byt_nr)
            ):
                self._instrument.write('DATA:ENC FASTEST; :WFMO:BYT_NR {}'
                                       .format(byt_nr))
                pre = self.preamble()
        # update setpoints etc.
        self.update(preamble=pre)
        # get raw data from device
        if pre['ENCDG'] == 'BIN':
            #order = dict('MSB':'>', 'LSB':'<')[pre['BYT_OR']]
            dtype = {'RI':{1:'b', 2:'h', 4:'i', 8:'q'}[pre['BYT_NR']], 
                     'RP':{1:'B', 2:'H', 4:'I', 8:'Q'}[pre['BYT_NR']], 
                     'FP':'f'}[pre['BN_FMT']]
            data = self._instrument._parent.visa_handle.query_binary_values(
                self.command+'?', datatype=dtype, container=np.array, 
                is_big_endian=(pre['BYT_OR']=='MSB'),
            )
        elif pre['ENCDG'] == 'ASC':
            converter = float if (pre['BN_FMT'] == 'FP') else int
            data = self._instrument._parent.visa_handle.query_ascii_values(
                self.command+'?', converter=converter, container=np.array
            )
        else:
            raise ValueError('Invalid encoding {}.'.format(pre['ENCDG']))
        # process data
        if not raw:
            if pre['PIXMAP']:
                # calculate mean value of the histogram in fast acquisitions 
                # and wfmdb modes -- use DPOPixmap to get the counts instead
                # pixmap is 1000 points times 252 bins
                data.shape = (1000, 252)
                ys = pre['YZERO'] + pre['YMULT']*(np.arange(252)-pre['YOFF'])
                data = (data * ys[None,:]).sum(1) / data.sum(1)
            else:
                # in all other modes shift and scale the data
                data = pre['YZERO'] + pre['YMULT']*data.astype(np.float32)
            if pre['PT_FMT'] == 'ENV':
                # in envelope mode, return min and max separately
                return tuple(data[offset::2].reshape(shape)
                             for offset, shape in enumerate(self.shapes))
        data.shape = self.shapes[0]
        return (data,)


class DPOPixmap(DPOCurve):
    def prepare(self):
        '''
        Prepare pixmap for data acquisition.

        Sets data start/stop on the instrument.
        Updates shape, label, units and setpoints via update().
        '''
        # set data source and range
        self._instrument.write('; '.join((
            'DATA:SOURCE {}'.format(self.source),
            'START {}'.format(1),
            'STOP {}'.format((1<<31)-1)
        )))
        self.update()

    def _update(self, pre):
        '''Update shape, label unit and setpoints from the waveform preamble.'''
        # determine label and data unit
        rows = 252
        columns = 1000
        if pre['NR_PT'] != rows*columns:
            raise ValueError('Pixmap size is not equal to rows*columns.')
        self.shapes = ((columns, rows),)
        self.labels = (pre['WFID'],)
        self.units = ('',)
        # determine setpoints
        sp_xs = DataArray(
            name='X', unit=pre['XUNIT'], is_setpoint=True, 
            preset_data=pre['XZERO'] + pre['XINCR']*np.arange(columns)
        )
        sp_ys = DataArray(
            name='Y', unit=pre['YUNIT'], is_setpoint=True,
            preset_data=pre['YZERO'] + pre['YMULT']*(np.arange(rows)-pre['YOFF'])
        )
        sp_ys.nest(len(sp_xs))
        self.setpoints = ((sp_xs, sp_ys),)

    def get_raw(self):
        return super().get_raw(raw=True)


class DPOChannel(InstrumentChannel):
    def __init__(self, parent, name, source):
        super().__init__(parent, name)
        self.source = source
        # vertical setup
        self.add_parameter(
            'state', label='State',
            get_cmd='SELECT:{}?'.format(source),
            set_cmd='SELECT:{} {}'.format(source, '{:d}'),
            get_parser=ibool, vals=vals.Bool()
        )
        self.add_parameter(
            'label', label='Label', 
            get_cmd='{}:LABEL:NAME?'.format(source),
            set_cmd='{}:LABEL:NAME {}'.format(source, '{}'),
            get_parser=qstr, vals=vals.Strings(0, 32)
        )
        # data transfer
        self.add_parameter('curve', DPOCurve, source=source)

    def show(self):
        self.state.set(True)

    def hide(self):
        self.state.set(False)


class DPOAuxiliaryChannel(InstrumentChannel):
    def __init__(self, parent, name, source):
        super().__init__(parent, name)
        self.source = source
        # vertical setup
        self.add_parameter(
            'bandwidth', label='Bandwidth', unit='Hz',
            get_cmd='{}:BANDWIDTH?'.format(source), get_parser=float,
            set_cmd='{}:BANDWIDTH {}'.format(source, '{:f}')
        )
        self.add_parameter(
            'coupling', label='Input Coupling',
            get_cmd='{}:COUPLING?'.format(source),
            set_cmd='{}:COUPLING {}'.format(source, '{}'),
            val_mapping={'ac':'AC', 'dc':'DC', 'dcreject':'DCREJ', 'gnd':'GND'}
        )
        self.add_parameter(
            'offset', label='Vertical Offset', unit='V',
            docstring='Vertical offset applied after digitizing, in volts.', 
            get_cmd='{}:OFFSET?'.format(source), 
            set_cmd='{}:OFFSET {}'.format(source, '{:f}'),
            get_parser=float
        )


class DPOAnalogChannel(DPOChannel):
    def __init__(self, parent, name, source):
        super().__init__(parent, name, source)
        # vertical setup
        self.add_parameter(
            'vposition', label='Vertical Position', unit='div',
            docstring='Vertical offset applied before digitizing, in divisions.', 
            get_cmd='{}:POSITION?'.format(source), 
            set_cmd='{}:POSITION {}'.format(source, '{:f}'),
            get_parser=float, vals=vals.Numbers(-8., 8.)
        )
        self.add_parameter(
            'vscale', label='Vertical Scale', unit='V', 
            get_cmd='{}:SCALE?'.format(source),
            set_cmd='{}:SCALE {}'.format(source, '{:f}')
        )
        self.add_parameter(
            'bandwidth', label='Bandwidth', unit='Hz',
            get_cmd='{}:BANDWIDTH?'.format(source), get_parser=float,
            set_cmd='{}:BANDWIDTH {}'.format(source, '{:f}')
        )
        self.add_parameter(
            'coupling', label='Input Coupling',
            get_cmd='{}:COUPLING?'.format(source),
            set_cmd='{}:COUPLING {}'.format(source, '{}'),
            val_mapping={'ac':'AC', 'dc':'DC', 'dcreject':'DCREJ', 'gnd':'GND'}
        )
        self.add_parameter(
            'offset', label='Vertical Offset', unit='V',
            docstring='Vertical offset applied after digitizing, in volts.', 
            get_cmd='{}:OFFSET?'.format(source), 
            set_cmd='{}:OFFSET {}'.format(source, '{:f}'),
            get_parser=float
        )
        self.add_parameter(
            'termination', label='Input Termination', unit='Ohm',
            get_cmd='{}:TERMINATION?'.format(source),
            set_cmd='{}:TERMINATION {}'.format(source, '{:f}'),
            get_parser=float, vals=vals.Enum(50, 50., 1000000, 1e6)
        )
        self.add_parameter(
            'deskew', label='Deskew', unit='s',
            get_cmd='{}:DESKEW?'.format(source), 
            set_cmd='{}:DESKEW {}'.format(source, '{:f}'), 
            get_parser=float, vals=vals.Numbers(-25e-9, 25e-9)
        )
        # data transfer
        self.add_parameter('pixmap', DPOPixmap, source=source)
        self.add_parameter('curvenext', DPOCurve, source=source, 
                           command='CURVENEXT')
        self.add_parameter('pixmapnext', DPOPixmap, source=source, 
                           command='CURVENEXT')


class DPOReferenceChannel(DPOChannel):
    def __init__(self, parent, name, source):
        super().__init__(parent, name, source)
        # display setup
        self.add_parameter(
            'hposition', label='Horizontal Position', unit='%',
            get_cmd='{}:HORIZONTAL:POSITION?'.format(source), 
            set_cmd='{}:HORIZONTAL:POSITION {}'.format(source, '{:f}'),
            get_parser=float, vals=vals.Numbers(0., 100.)
        )
        self.add_parameter(
            'vposition', label='Vertical Position', unit='div',
            get_cmd='{}:VERTICAL:POSITION?'.format(source), 
            set_cmd='{}:VERTICAL:POSITION {}'.format(source, '{:f}'),
            get_parser=float, vals=vals.Numbers(-8., 8.)
        )
        self.add_parameter(
            'vscale', label='Vertical Scale', unit='V/A/W', 
            get_cmd='{}:VERTICAL:SCALE?'.format(source),
            set_cmd='{}:VERTICAL:SCALE {}'.format(source, '{:f}')
        )

    def autoscale(self):
        self._instrument.write('{}:VERTICAL:AUTOSCALE'.format(self.source))


class DPOMathChannel(DPOReferenceChannel):
    def __init__(self, parent, name, source):
        super().__init__(parent, name, source)
        # expression setup
        self.add_parameter(
            'expression', label='Expression', 
            docstring='Mathematical expression used to calculate the data', 
            get_cmd='{}:DEFINE?'.format(self.source), get_parser=qstr, 
            set_cmd='{}:DEFINE "{}"'.format(self.source, '{}'),
        )
        self.add_parameter(
            'averages', label='Averages',
            docstring='''
            Number of averages after which exponential averaging will be used
            instead of stable averaging. Has no effect unless AVG() function 
            is part of expression.
            ''',
            get_cmd='{}:NUMAVG?'.format(self.source),
            set_cmd='{}:NUMAVG {}'.format(self.source, '{:d}'),
            get_parser=int, vals=vals.Ints(1)
        )
        # display setup
        del self.parameters['hposition']


class DPOTrigger(InstrumentChannel):
    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        self.channel = channel
        # common settings
        self.add_parameter(
            'level', label='Trigger level', unit='V', 
            get_cmd='TRIG:{}:LEVEL?'.format(channel),
            set_cmd='TRIG:{}:LEVEL {}'.format(channel, '{:f}'), 
            get_parser=float, vals=vals.Numbers()
        )
        self.add_parameter(
            'type', label='Trigger type',
            get_cmd='TRIG:{}:TYPE?'.format(channel),
            set_cmd='TRIG:{}:TYPE {}'.format(channel, '{}'),
            val_mapping={'edge':'EDGE', 'logic':'LOGI', 'pulse':'PUL', 
                         'video':'VID', 'i2c':'I2C', 'can':'CAN', 'spi':'SPI',
                         'comm':'COMM', 'serial':'SERIAL', 'rs232':'RS232'}
        )
        self.add_parameter(
            'ready', label='Trigger ready state', 
            get_cmd='TRIG:{}:READY?'.format(channel), get_parser=ibool
        )
        # A trigger only settings
        if channel == 'A':
            self.add_parameter(
                'mode', label='Trigger mode', 
                get_cmd='TRIG:A:MODE?', set_cmd='TRIG:A:MODE {}', 
                val_mapping={'auto':'AUTO', 'normal':'NORM'}
            )
            self.add_parameter(
                'holdoff_type', label='Trigger holdoff type', 
                get_cmd='TRIG:A:HOLDOFF:BY?', set_cmd='TRIG:A:HOLDOFF:BY {}', 
                val_mapping={'time':'TIM', 'default':'DEF', 'random':'RAND', 
                             'auto':'AUTO'}
            )
            self.add_parameter(
                'holdoff_time', label='Trigger holdoff time', unit='s', 
                get_cmd='TRIG:A:HOLDOFF:TIME?', get_parser=float,
                set_cmd='TRIG:A:HOLDOFF:TIME {:f}', vals=vals.Numbers(0., 12.)
            )
        # B trigger only settings
        if channel == 'B':
            self.add_parameter(
                'state',
                docstring='Is the B trigger is part of the triggering sequence?', 
                get_cmd='TRIG:B:STATE?', get_parser=ibool,
                set_cmd='TRIG:B:STATE {:d}', vals=vals.Bool()
            )
            self.add_parameter(
                'by',
                docstring='Selects whether B trigger occurs after a specified '
                          'number of events or a specified time after A.', 
                get_cmd='TRIG:B:BY?', set_cmd='TRIG:B:BY {}',
                val_mapping={'events':'EVENTS', 'time':'TIM'}
            )
            self.add_parameter(
                'events',
                docstring='Number of B trigger events before acquisition.', 
                get_cmd='TRIG:B:EVENTS:COUNT?', get_parser=int, 
                set_cmd='TRIG:B:EVENTS:COUNT {:d}', vals=vals.Ints(1, 10000000)
            )
            self.add_parameter(
                'time', unit='s', 
                get_cmd='TRIG:B:TIME?', get_parser=float,
                set_cmd='TRIG:B:TIME {:f}', vals=vals.Numbers(0.)
            )
        # edge trigger
        self.add_parameter(
            'edge_coupling', 
            get_cmd='TRIG:{}:EDGE:COUPLING?'.format(channel),
            set_cmd='TRIG:{}:EDGE:COUPLING {}'.format(channel, '{}'),
            val_mapping={'ac':'AC', 'dc':'DC', 'lf-reject':'LFR', 
                         'hf-reject':'HFR', 'noise-reject':'NOISE'}
        )
        self.add_parameter(
            'edge_slope', label='Edge trigger polarity', 
            get_cmd='TRIG:{}:EDGE:SLOPE?'.format(channel),
            set_cmd='TRIG:{}:EDGE:SLOPE {}'.format(channel, '{}'),
            val_mapping={'rising':'RIS', 'falling':'FALL', 'either':'EIT'}
        )
        self.add_parameter(
            'edge_source', label='Edge trigger source',
            get_cmd='TRIG:{}:EDGE:SOURCE?'.format(channel),
            set_cmd='TRIG:{}:EDGE:SOURCE {}'.format(channel, '{}'),
            val_mapping={'aux':'AUX', 'line':'LINE', 
                         'ch1':'CH1', 'ch2':'CH2', 'ch3':'CH3', 'ch4':'CH4'}
        )

    def autoset(self):
        self.write('TRIGGER:{} SETLEVEL'.format(self.channel))


class Tektronix_DPO70000(VisaInstrument):
    def __init__(self, name, address):
        super().__init__(name, address, terminator='\n')

        # Acquisition setup
        self.add_parameter(
            'acq_state', label='Acuqisition state',
            get_cmd='ACQ:STATE?', get_parser=ibool,
            set_cmd='ACQ:STATE {:d}', vals=vals.Bool()
        )
        self.add_parameter(
            'acq_single', label='Single acquisition mode',
            get_cmd='ACQ:STOPAFTER?', set_cmd='ACQ:STOPAFTER {}', 
            val_mapping={False:'RUNST', True:'SEQ'}
        )
        self.add_parameter(
            'acq_mode', label='Acquisition Mode',
            docstring='''
            In each each acquisition interval, show
             * sample: the first sampled value
             * peakdetect: the minimum and maximum sample values
             * hires: the average of all samples
             * average: the first sample averaged over separate acquisitions
                        the instrument returns a running exponential average
             * wfmdb: a histogram of all samples of one or more acquisitions
             * envelope: the minimum and maximum sample values of multiple 
                         acquisitions
            ''',
            get_cmd='ACQ:MODE?', set_cmd='ACQ:MODE {}',
            val_mapping={'sample':'SAM', 'peakdetect':'PEAK', 'hires':'HIR',
                         'average':'AVE', 'wfmdb':'WFMDB', 'envelope':'ENV'}
        )
        #self.add_parameter(
        #    'acq_mode_actual', label='Actual Acquisition Mode',
        #    get_cmd='ACQ:MODE:ACTUAL?'
        #)
        self.add_parameter(
            'acq_numacq', label='Number of acquisitions', 
            docstring='Total number of acquisitions since last run command.'
                      'Counting stops when 2^30-1 is reached.', 
            get_cmd='ACQ:NUMACQ?', get_parser=int
        )
        self.add_parameter(
            'acq_fast', label='Fast acquisitions',
            docstring='''
            Staus of fast acquisitions.
            Fast acquisitions always use acq_mode==sample.
            ''', 
            get_cmd='FASTACQ:STATE?', get_parser=ibool,
            set_cmd='FASTACQ:STATE {:d}; HIACQRATE 1', vals=vals.Bool()
        )
        self.add_parameter(
            'acq_averages', label='Number of averages',
            get_cmd='ACQ:NUMAVG?', get_parser=int,
            set_cmd='ACQ:NUMAVG {:d}', vals=vals.Ints(1)
        )
        self.add_parameter(
            'acq_envelopes', label='Number of envelope waveforms',
            get_cmd='ACQ:NUMENV?', get_parser=int,
            set_cmd='ACQ:NUMENV {:d}', vals=vals.Ints(1, 2000000000)
        )
        self.add_parameter(
            'acq_wfmdbs', label='Number of wfmdb samples',
            docstring='Total number of counts in the WFMDB pixmap.',
            get_cmd='ACQ:NUMSAMPLES?', get_parser=int,
            set_cmd='ACQ:NUMSAMPLES {:d}', vals=vals.Ints(5000, 2147400000)
        )
        self.add_parameter(
            'acq_sampling', label='Sampling mode', 
            get_cmd='ACQ:SAMPLINGMODE?', set_cmd='ACQ:SAMPINGMODE {}', 
            val_mapping={'realtime':'RT', 'equivalent':'ET', 'interpolated':'IT'}
        )

        # Horizontal setup
        self.add_parameter(
            'hmode', label='Horizontal Mode',
            docstring='Selects the automatic horzontal model.'
                      'auto: Set time/division. Keeps the record length constant.'
                      'constant: Set time/division. Keeps the sample rate constant.'
                      'manual: Set record length and sample rate.', 
            get_cmd='HOR:MODE?', set_cmd='HOR:MODE {}', 
            val_mapping={'auto':'AUTO', 'constant':'CONS', 'manual':'MAN'}
        )
        self.add_parameter(
            'hscale', label='Horizontal Scale', unit='s', 
            docstring='Horizontal scale in seconds per division.', 
            get_cmd='HOR:MODE:SCALE?', get_parser=float,
            set_cmd='HOR:MODE:SCALE {}'
        )
        self.add_parameter(
            'samplerate', label='Sample Rate', unit='1/s',
            get_cmd='HOR:MODE:SAMPLERATE?', get_parser=float,
            set_cmd='HOR:MODE:SAMPLERATE {}', set_parser=float
        )
        self.add_parameter(
            'recordlength', label='Record Length', 
            get_cmd='HOR:MODE:RECORDLENGTH?', get_parser=int,
            set_cmd='HOR:MODE:RECORDLENGTH {}', set_parser=int
        )
        self.add_parameter(
            'hposition', label='Horizontal Position', unit='%', 
            docstring='Position of the trigger point on screen in %.',
            get_cmd='HOR:POS?', get_parser=float,
            set_cmd='HOR:POS {}', vals=vals.Numbers(0., 100.)
        )
        self.add_parameter(
            'hdelay_status', label='Horizontal Delay Status',
            get_cmd='HOR:DELAY:MODE?', get_parser=ibool,
            set_cmd='HOR:DELAY:MODE {:d}', vals=vals.Bool()
        )
        self.add_parameter(
            'hdelay_pos', label='Horizontal Position', unit='%', 
            docstring='Position of the trigger point on screen in %.',
            get_cmd='HOR:DELAY:POS?', get_parser=float,
            set_cmd='HOR:DELAY:POS {}', vals=vals.Numbers(0., 100.)
        )
        self.add_parameter(
            'hdelay_time', label='Horizontal Delay', unit='s',
            get_cmd='HOR:DELAY:TIME?', get_parser=float,
            set_cmd='HOR:DELAY:TIME {}'
        )

        # Fast frame setup
        self.add_parameter(
            'frame_state', label='Fast frame acquisition state',
            get_cmd='HOR:FAST:STATE?', get_parser=ibool,
            set_cmd='HOR:FAST:STATE {:d}', vals=vals.Bool()
        )
        self.add_parameter(
            'frame_count', label='Fast frame count',
            get_cmd='HOR:FAST:COUNT?', get_parser=int,
            set_cmd='HOR:FAST:COUNT {:d}', vals=vals.Ints(1)
        )
        self.add_parameter(
            'frame_maxcount', label='Fast frame maximum count',
            get_cmd='HOR:FAST:MAXFRAMES?', get_parser=int
        )
        self.add_parameter(
            'frame_seqstop', label='Fast frame sequence stop condition',
            docstring='Fast frame single-sequence mode stop condition. '
                      'Stops after N frames in `single`, manually otherwise.', 
            get_cmd='HOR:FAST:SEQ?', set_cmd='HOR:FAST:SEQ {}',
            val_mapping={'single': 'FIR', 'manual':'LAST'}
        )
        self.add_parameter(
            'frame_numacq', label='Fast frame number of frames acquired',
            get_cmd='ACQ:NUMFRAMESACQUIRED?', get_parser=int
        )
        self.add_parameter(
            'frame_sumframe', label='Fast frame summary frame', 
            get_cmd='HOR:FAST:SUMFRAME?', set_cmd='HOR:FAST:SUMFRAME {}', 
            val_mapping={'none': 'NON', 'average':'AVE', 'envelope':'ENV'}
        )
        #self.add_parameter(
        #    'frame_16bit', label='Fast frame 16bit mode',
        #    docstring='If True, the averaged summary frame has 16bit resolution.', 
        #    get_cmd='HOR:FAST:SIXTEENBIT?', get_parser=ibool, 
        #    set_cmd='HOR:FAST:SIXTEENBIT {:d}', vals=vals.Bool()
        #)

        # Triggers
        self.add_submodule('triggerA', DPOTrigger(self, 'triggerA', 'A'))
        self.add_submodule('triggerB', DPOTrigger(self, 'triggerB', 'B'))

        # Channels
        for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
            self.add_submodule(ch, DPOAnalogChannel(self, ch, ch.upper()))
        for ch in ['math1', 'math2', 'math3', 'math4']:
            self.add_submodule(ch, DPOMathChannel(self, ch, ch.upper()))
        for ch in ['ref1', 'ref2', 'ref3', 'ref4']:
            self.add_submodule(ch, DPOReferenceChannel(self, ch, ch.upper()))
        self.add_submodule('auxin', DPOAuxiliaryChannel(self, 'auxin', 'AUXIN'))

        # *IDN?
        self.connect_message()

    def autoset(self):
        self.write('AUTOSET')

    def clear(self):
        '''Clear all acquisitions, measurements and waveforms.'''
        self.write('CLEAR')

    def run(self):
        '''Start acquisition in continuous (run/stop) mode.'''
        self.acq_single.set(False)
        self.acq_state.set(True)

    def single(self, wait=True, timeout=None):
        '''Start acquisition in single mode. Optionally call wait().'''
        self.acq_single.set(True)
        self.acq_state.set(True)
        if wait:
            return self.wait(timeout)

    def start(self):
        '''Start acquisition in current mode.'''
        self.acq_state.set(True)

    def stop(self):
        '''Stop acquisition.'''
        self.acq_state.set(False)

    def trigger(self):
        '''Force a trigger event.'''
        self.write('TRIGGER')

    def wait(self, timeout=None):
        '''
        Wait for acquisition to finish. Requires single acquisition mode.

        Input
        -----
        timeout: `float`, default self.timeout()
            Maximum time to wait.

        Return
        ------
        True if acquisition finished before the timeout, False otherwise.
        '''
        if timeout is None:
            timeout = self.timeout.get()
        max_time = time.time() + timeout
        while (time.time() < max_time):
            if not self.acq_state():
                return True
        return False