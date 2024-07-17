#!/usr/bin/env python3

from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from astromodels.functions.priors import Log_uniform_prior, Uniform_prior, Cosine_Prior


class BALROGLikePositionPrior(BALROGLike):
    """
    BALROG Like but with a more constrained prior on the Position
    Not for production use!
    """

    def __init__(
        self,
        name,
        observation,
        drm_generator=None,
        background=None,
        time=0,
        free_position=True,
        verbose=True,
        use_cache=False,
        swift_position=None,
        **kwargs,
    ):
        self._swift_position = swift_position
        if self._swift_position is not None:
            self._swift_ra = swift_position.ra.deg
            self._swift_dec = swift_position.dec.deg
        self._free_position = free_position
        self._use_cache = use_cache
        if drm_generator is None:
            # If a generator is not supplied
            # then make sure that there is a
            # balrog response

            assert isinstance(
                observation.response, BALROG_DRM
            ), "The response associated with the observation is not a BALROG"

        else:
            # here we will reset the response
            # this is violating the fact that
            # the response is provate

            balrog_drm = BALROG_DRM(drm_generator, 0.0, 0.0)

            observation._response = balrog_drm

        super(BALROGLike, self).__init__(
            name, observation, background, verbose, **kwargs
        )

        # only on the start up

        self._response.set_time(time)

    def set_model(self, likelihoodModel):
        """
        Set the model and free the location parameters


        :param likelihoodModel:
        :return: None
        """

        # set the standard likelihood model

        super(BALROGLikePositionPrior, self).set_model(likelihoodModel)

        # now free the position
        # if it is

        for key in self._like_model.point_sources.keys():
            self._like_model.point_sources[key].position.ra.free = True
            self._like_model.point_sources[key].position.dec.free = True
            if self._swift_position is not None:
                self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                    lower_bound=float(self._swift_ra) - 10,
                    upper_bound=float(self._swift_ra) + 10,
                )
                self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                    lower_bound=float(self._swift_dec) - 10,
                    upper_bound=float(self._swift_dec) + 10,
                )
                print(f"Set priors according to Swfit position +/- 10 deg")
            else:
                self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                    lower_bound=0,
                    upper_bound=360,
                )
                self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                    lower_bound=-90,
                    upper_bound=90,
                )

            ra = self._like_model.point_sources[key].position.ra.value
            dec = self._like_model.point_sources[key].position.dec.value

        self._response.set_location(ra, dec, cache=self._use_cache)

    def get_model(self, precalc_fluxes=None):
        # Here we update the GBM drm parameters which creates and new DRM for that location
        # we should only be dealing with one source for GBM

        # update the location

        # assumes that the is only one point source which is how it should be!
        ra, dec = self._like_model.get_point_source_position(0)

        self._response.set_location(ra, dec, cache=self._use_cache)

        return super(BALROGLike, self).get_model(precalc_fluxes)

    @classmethod
    def from_spectrumlike(
        cls,
        spectrum_like,
        time,
        drm_generator=None,
        free_position=True,
        swift_position=None,
    ):
        """
        Generate a BALROGlike from an existing SpectrumLike child


        :param spectrum_like: the existing spectrumlike
        :param time: the time to generate the RSPs at
        :param drm_generator: optional BALROG DRM generator
        :param free_position: if the position should be free
        :return:
        """

        return cls(
            spectrum_like.name,
            spectrum_like._observed_spectrum,
            drm_generator,
            spectrum_like._background_spectrum,
            time=time,
            free_position=free_position,
            verbose=spectrum_like._verbose,
            swift_position=swift_position,
        )
